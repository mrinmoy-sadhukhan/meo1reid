import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import ops
import torch
from utils.inference import run_inferencev1_0
import cv2
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import os
import time  # Add this import

# Default colors for visualization of boxes
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]
COLORS *= 100  # Repeat colors to cover all classes


class DETRBoxVisualizer:
    def __init__(self, class_labels, empty_class_id, normalization_params=(None, None)):
        """
        The DETR box visualizer is responsible for visualizing the inputs/outputs of the DETR model.

        You can use the public API of the class to:
        - Visualize a single image or inference results with "visualize_image()"
        - Visualize batch inference results using a validation dataset with "visualize_validation_inference()"
        Args:
            class_labels (list): List of class labels.
            normalization_params (tuple): Mean and standard deviation used for normalization.
            empty_class_id (int): The class ID representing 'no object'.
        """
        self.class_labels = class_labels
        self.empty_class_id = empty_class_id
        self.class_to_color = {}

        if normalization_params != (None, None) and type(normalization_params) == tuple:
            if len(normalization_params) != 2:
                raise ValueError(
                    "Expected normalization_params to be a tuple of length 2!"
                )

            mean, std = normalization_params
            if len(mean) != 3 or len(std) != 3:
                raise ValueError("Expected mean and std to be tuples of length 3!")
            self.normalization_params = normalization_params
        else:
            # Assume ImageNet normalization
            self.normalization_params = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the unnormalize transform
        mean, std = self.normalization_params
        self.unnormalize = T.Normalize(
            mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
        )

    def _revert_normalization(self, tensor):
        """
        Reverts the normalization of an image tensor.

        Args:
            tensor (torch.Tensor): Normalized image tensor.

        Returns:
            torch.Tensor: Denormalized image tensor.
        """
        return self.unnormalize(tensor)

    def _visualize_image(
        self, im, boxes, class_ids, scores=None, ax=None, show_scores=True
    ):
        """
        Visualizes a single image with bounding boxes and predicted probabilities.
        NOTE: The boxes tensors is expected to be in the format (xmin, ymin, xmax, ymax) and
              in pixel space already (not normalized).

        Args:
            im (np.array): Image to visualize.
            boxes (np.array): Bounding boxes.
            class_ids (np.array): Class IDs for each box.
            scores (np.array, optional): Probabilities for each box.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis object.
            show_scores (bool, optional): Whether to show the predicted probabilities.
        """
        if ax is None:
            ax = plt.gca()

        # Revert normalization for image
        im = self._revert_normalization(im).permute(1, 2, 0).cpu().clip(0, 1)

        ax.imshow(im)
        ax.axis("off")  # Hide axes

        for i, b in enumerate(boxes.tolist()):
            xmin, ymin, xmax, ymax = b

            if scores is not None:
                score = scores[i]
            else:
                score = None

            if class_ids is not None:
                cl = class_ids[i]
            else:
                raise ValueError("No class IDs provided for visualization!")

            # Assign a color to the class if not already assigned
            if cl not in self.class_to_color:
                self.class_to_color[cl] = COLORS[cl % len(COLORS)]

            color = self.class_to_color[cl]

            # Draw bounding box
            patch = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color=color,
                linewidth=2,
            )
            ax.add_patch(patch)

            # Add label text
            text = (
                f"{self.class_labels[cl]}"
                if score is None or not show_scores
                else f"{self.class_labels[cl]}: {score:0.2f}"
            )
            ax.text(
                xmin, ymin, text, fontsize=7, bbox=dict(facecolor="yellow", alpha=0.5)
            )

    def visualize_video_inference_v1_0(
            self,
            model,
            video_path,
            save_dir,
            image_size=480,
            batch_size=5,
            nms_threshold=0.3,
            show_timestamp=True,  # overlay time text
            start_time=0.0,        # start (seconds)
            end_time=None,         # end (seconds)
        ):
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        print(f"Total frames: {total_frames}, FPS: {fps}, Duration: {duration:.2f}s")

        # Validate times
        if start_time < 0 or (end_time and end_time <= start_time):
            raise ValueError("Invalid start_time or end_time range.")

        # Skip frames before start_time
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Starting processing from {start_time:.2f}s (frame {start_frame})")

        # Compute last frame to process
        if end_time is not None and end_time < duration:
            stop_frame = int(end_time * fps)
            print(f"Ending processing at {end_time:.2f}s (frame {stop_frame})")
        else:
            stop_frame = total_frames

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.normalization_params[0], std=self.normalization_params[1]),
            T.Resize((image_size, image_size), antialias=True),
        ])

        frames, frame_batches, processed_frames = [], [], []

        print(f"Running inference on device: {self.device}")

        # FPS calculation variables
        total_processing_time = 0.0
        total_inference_time = 0.0
        total_frames_processed = 0
        start_processing_time = time.time()

        while cap.isOpened():
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame >= stop_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Preprocessing time
            preprocess_start = time.time()
            img_tensor = transform(pil_img).unsqueeze(0)
            preprocess_time = time.time() - preprocess_start

            video_h, video_w, _ = frame.shape
            frames.append(frame_rgb)
            frame_batches.append(img_tensor)

            # Run inference batch
            if len(frame_batches) == batch_size:
                batch_input = torch.cat(frame_batches, dim=0)

                # Inference time measurement
                inference_start = time.time()
                inference_results = run_inferencev1_0(
                    model=model,
                    device=self.device,
                    inputs=batch_input,
                    nms_threshold=nms_threshold,
                    image_size=image_size,
                    empty_class_id=self.empty_class_id,
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                # Visualization and post-processing
                postprocess_start = time.time()
                for i in range(batch_size):
                    nms_boxes, nms_probs, nms_classes = inference_results[i]
                    
                    if nms_boxes.size == 0:
                        frame_to_save = frames[i]
                    else:
                        fig, ax = plt.subplots(figsize=(image_size / 100, image_size / 100), dpi=100)
                        ax.set_frame_on(False)
                        ax.set_axis_off()
                        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                        self._visualize_image(batch_input[i].cpu(), nms_boxes, nms_classes, nms_probs, ax=ax)
                        fig.canvas.draw()
                        frame_to_save = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
                        frame_to_save = cv2.resize(frame_to_save, (video_w, video_h))
                        plt.close(fig)

                    # Add timestamp overlay
                    if show_timestamp:
                        timestamp_sec = (start_frame + total_frames_processed + i) / fps
                        timestamp_str = f"{int(timestamp_sec//60):02d}:{int(timestamp_sec%60):02d}"
                        cv2.putText(
                            frame_to_save,
                            timestamp_str,
                            (20, video_h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    processed_frames.append(frame_to_save)

                postprocess_time = time.time() - postprocess_start
                total_processing_time += (preprocess_time + inference_time + postprocess_time)
                total_frames_processed += batch_size

                frames, frame_batches = [], []

        # Process remaining frames in the last batch
        if frame_batches:
            batch_input = torch.cat(frame_batches, dim=0)
            
            inference_start = time.time()
            inference_results = run_inferencev1_0(
                model=model,
                device=self.device,
                inputs=batch_input,
                nms_threshold=nms_threshold,
                image_size=image_size,
                empty_class_id=self.empty_class_id,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            inference_time = time.time() - inference_start
            total_inference_time += inference_time

            postprocess_start = time.time()
            for i in range(len(frame_batches)):
                nms_boxes, nms_probs, nms_classes = inference_results[i]
                
                if nms_boxes.size == 0:
                    frame_to_save = frames[i]
                else:
                    fig, ax = plt.subplots(figsize=(image_size / 100, image_size / 100), dpi=100)
                    ax.set_frame_on(False)
                    ax.set_axis_off()
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    self._visualize_image(batch_input[i].cpu(), nms_boxes, nms_classes, nms_probs, ax=ax)
                    fig.canvas.draw()
                    frame_to_save = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
                    frame_to_save = cv2.resize(frame_to_save, (video_w, video_h))
                    plt.close(fig)

                # Add timestamp overlay
                if show_timestamp:
                    timestamp_sec = (start_frame + total_frames_processed + i) / fps
                    timestamp_str = f"{int(timestamp_sec//60):02d}:{int(timestamp_sec%60):02d}"
                    cv2.putText(
                        frame_to_save,
                        timestamp_str,
                        (20, video_h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                processed_frames.append(frame_to_save)

            postprocess_time = time.time() - postprocess_start
            total_processing_time += (preprocess_time + inference_time + postprocess_time)
            total_frames_processed += len(frame_batches)

        cap.release()
        total_elapsed_time = time.time() - start_processing_time

        # === Calculate and display FPS statistics ===
        print("\n" + "="*50)
        print("PERFORMANCE STATISTICS")
        print("="*50)
        
        if total_frames_processed > 0:
            # Overall FPS (total processing time)
            overall_fps = total_frames_processed / total_elapsed_time
            print(f"📊 Overall FPS: {overall_fps:.2f}")
            
            # Inference-only FPS
            inference_fps = total_frames_processed / total_inference_time if total_inference_time > 0 else 0
            print(f"🧠 Inference FPS: {inference_fps:.2f}")
            
            # Processing FPS (including pre/post-processing)
            processing_fps = total_frames_processed / total_processing_time if total_processing_time > 0 else 0
            print(f"⚡ Processing FPS: {processing_fps:.2f}")
            
            # Time breakdown
            print(f"⏱️  Total time: {total_elapsed_time:.2f}s")
            print(f"🔍 Inference time: {total_inference_time:.2f}s ({total_inference_time/total_elapsed_time*100:.1f}%)")
            print(f"🎨 Visualization time: {total_processing_time - total_inference_time:.2f}s ({(total_processing_time - total_inference_time)/total_elapsed_time*100:.1f}%)")
            print(f"📈 Frames processed: {total_frames_processed}")
        else:
            print("❌ No frames were processed")

        # === Save video only for the selected time range ===
        os.makedirs(save_dir, exist_ok=True)
        if end_time:
            fname = f"processed_{int(start_time)}s_to_{int(end_time)}s.mp4"
        else:
            fname = f"processed_from_{int(start_time)}s.mp4"

        output_video_path = os.path.join(save_dir, fname)

        if processed_frames:
            # Measure video writing time
            video_write_start = time.time()
            clip = ImageSequenceClip(processed_frames, fps=fps)
            clip.write_videofile(output_video_path, codec="libx264", logger=None)
            video_write_time = time.time() - video_write_start
            
            print(f"💾 Video saved to: {output_video_path}")
            print(f"📹 Video writing time: {video_write_time:.2f}s")
        else:
            print("⚠️ No processed frames to save")

        return {
            'overall_fps': overall_fps,
            'inference_fps': inference_fps,
            'processing_fps': processing_fps,
            'total_frames': total_frames_processed,
            'total_time': total_elapsed_time,
            'output_path': output_video_path if processed_frames else None
        }
    
    def _fast_visualize_image(self, frame_rgb, boxes, class_ids, scores, original_size):
        """Fast visualization using OpenCV instead of matplotlib"""
        h, w = original_size
        frame_vis = frame_rgb.copy()
        
        for i, (box, cls_id, score) in enumerate(zip(boxes, class_ids, scores)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Assign color
            color = COLORS[cls_id % len(COLORS)]
            color_rgb = [int(c * 255) for c in color][::-1]  # Convert to BGR
            
            # Draw bounding box
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color_rgb, 2)
            
            # Draw label
            label = f"{self.class_labels[cls_id]} {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(frame_vis, (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), color_rgb, -1)
            cv2.putText(frame_vis, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame_vis
    
    def visualize_video_inference_optimized(
        self,
        model,
        video_path,
        save_dir,
        image_size=480,
        batch_size=8,
        nms_threshold=0.3,
        show_timestamp=True,
        start_time=0.0,
        end_time=None,
        use_fp16=True,
        use_fast_vis=True,
    ):
    
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        stop_frame = int(end_time * fps) if end_time else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Model optimization
        if use_fp16 and self.device.type == "cuda":
            #model = model.half()
            dtype = torch.float32
        else:
            dtype = torch.float32

        model.eval()
        
        # Transform
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.normalization_params[0], std=self.normalization_params[1]),
            T.Resize((image_size, image_size), antialias=True),
        ])
        
        # Add dtype conversion if using FP16
        if use_fp16 and self.device.type == "cuda":
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=self.normalization_params[0], std=self.normalization_params[1]),
                T.Resize((image_size, image_size), antialias=True),
                T.ConvertImageDtype(dtype),
            ])

        frames, frame_batches, processed_frames = [], [], []
        original_sizes = []

        # Performance tracking
        total_frames_processed = 0
        total_inference_time = 0.0
        start_time_total = time.time()

        print(f"🚀 Running optimized inference on: {self.device}")

        while cap.isOpened():
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame >= stop_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Fast preprocessing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            img_tensor = transform(pil_img).unsqueeze(0)
            
            video_h, video_w, _ = frame.shape
            frames.append(frame_rgb)
            frame_batches.append(img_tensor)
            original_sizes.append((video_h, video_w))

            # Process batch
            if len(frame_batches) == batch_size:
                batch_input = torch.cat(frame_batches, dim=0).to(self.device)

                # Inference
                inference_start = time.time()
                with torch.no_grad():
                    inference_results = run_inferencev1_0(
                        model=model,
                        device=self.device,
                        inputs=batch_input,
                        nms_threshold=nms_threshold,
                        image_size=image_size,
                        empty_class_id=self.empty_class_id,
                    )
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                # Fast visualization
                for i in range(batch_size):
                    nms_boxes, nms_probs, nms_classes = inference_results[i]
                    original_size = original_sizes[i]
                    
                    if use_fast_vis:
                        # Use fast OpenCV visualization
                        if len(nms_boxes) > 0:  # FIX: Use len() instead of .size
                            # Scale boxes to original size
                            scale_x = original_size[1] / image_size
                            scale_y = original_size[0] / image_size
                            scaled_boxes = nms_boxes * [scale_x, scale_y, scale_x, scale_y]
                            frame_vis = self._fast_visualize_image(
                                frames[i], scaled_boxes, nms_classes, nms_probs, original_size
                            )
                        else:
                            frame_vis = frames[i]
                    else:
                        # Fallback to matplotlib
                        frame_vis = self._matplotlib_visualize(
                            batch_input[i].cpu(), nms_boxes, nms_classes, nms_probs, original_size
                        )

                    # Add timestamp
                    if show_timestamp:
                        timestamp_sec = (start_frame + total_frames_processed + i) / fps
                        timestamp_str = f"{int(timestamp_sec//60):02d}:{int(timestamp_sec%60):02d}"
                        cv2.putText(
                            frame_vis, timestamp_str, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
                        )

                    processed_frames.append(frame_vis)

                total_frames_processed += batch_size
                frames, frame_batches, original_sizes = [], [], []

        # Process remaining frames
        if frame_batches:
            batch_input = torch.cat(frame_batches, dim=0).to(self.device)
            
            inference_start = time.time()
            with torch.no_grad():
                inference_results = run_inferencev1_0(
                    model=model, device=self.device, inputs=batch_input,
                    nms_threshold=nms_threshold, image_size=image_size,
                    empty_class_id=self.empty_class_id,
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            for i in range(len(frame_batches)):
                nms_boxes, nms_probs, nms_classes = inference_results[i]
                original_size = original_sizes[i]
                
                if use_fast_vis:
                    if len(nms_boxes) > 0:  # FIX: Use len() here too
                        scale_x = original_size[1] / image_size
                        scale_y = original_size[0] / image_size
                        scaled_boxes = nms_boxes * [scale_x, scale_y, scale_x, scale_y]
                        frame_vis = self._fast_visualize_image(
                            frames[i], scaled_boxes, nms_classes, nms_probs, original_size
                        )
                    else:
                        frame_vis = frames[i]
                else:
                    frame_vis = self._matplotlib_visualize(
                        batch_input[i].cpu(), nms_boxes, nms_classes, nms_probs, original_size
                    )

                if show_timestamp:
                    timestamp_sec = (start_frame + total_frames_processed + i) / fps
                    timestamp_str = f"{int(timestamp_sec//60):02d}:{int(timestamp_sec%60):02d}"
                    cv2.putText(
                        frame_vis, timestamp_str, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
                    )

                processed_frames.append(frame_vis)

            total_frames_processed += len(frame_batches)

        cap.release()
        total_time = time.time() - start_time_total

        # Save results
        self._save_video_results(processed_frames, save_dir, fps, start_time, end_time)
        
        # Print performance
        self._print_performance_stats(total_frames_processed, total_time, total_inference_time)
        
        #return processed_frames

    def _save_video_results(self, processed_frames, save_dir, fps, start_time, end_time):
        """Save processed video"""
        os.makedirs(save_dir, exist_ok=True)
        if end_time:
            fname = f"processed_{int(start_time)}s_to_{int(end_time)}s.mp4"
        else:
            fname = f"processed_from_{int(start_time)}s.mp4"

        output_video_path = os.path.join(save_dir, fname)

        if processed_frames:
            clip = ImageSequenceClip(processed_frames, fps=fps)
            clip.write_videofile(output_video_path, codec="libx264", logger=None)
            print(f"💾 Video saved to: {output_video_path}")

    def _print_performance_stats(self, total_frames, total_time, inference_time):
        """Print performance statistics"""
        print("\n" + "="*50)
        print("PERFORMANCE STATISTICS")
        print("="*50)
        
        if total_frames > 0:
            overall_fps = total_frames / total_time
            inference_fps = total_frames / inference_time if inference_time > 0 else 0
            
            print(f"📊 Overall FPS: {overall_fps:.2f}")
            print(f"🧠 Inference FPS: {inference_fps:.2f}")
            print(f"⏱️  Total time: {total_time:.2f}s")
            print(f"🔍 Inference time: {inference_time:.2f}s ({inference_time/total_time*100:.1f}%)")
            #print(f"📈 Frames processed: {total_frames}")
        else:
            print("❌ No frames were processed")
    
    def _create_preallocated_batch(self, batch_size, image_size, dtype=torch.float32):
        """Pre-allocate batch tensor to avoid repeated allocation"""
        return torch.empty((batch_size, 3, image_size, image_size), dtype=dtype, device=self.device)
    
    def inference_with_streams(self, model, batch_input, stream):
        """Use CUDA streams for parallel inference"""
        with torch.cuda.stream(stream):
            return run_inferencev1_0(model, self.device, batch_input, ...)