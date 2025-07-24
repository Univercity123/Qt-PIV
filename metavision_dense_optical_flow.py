# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Code sample showing how to use Metavision SDK to display results of dense optical flow.
代码示例展示了如何使用 Metavision SDK 显示密集光流的结果。
"""

import numpy as np
np.float = np.float64
np.int = np.int_
import os
import h5py
import math
import cv2
from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_sdk_cv import PlaneFittingFlowAlgorithm, TimeGradientFlowAlgorithm, TripletMatchingFlowAlgorithm, \
    DenseFlowFrameGeneratorAlgorithm, SpatioTemporalContrastAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIAction, UIKeyEvent
# Temporary solution to fix the numpy deprecated alias in skvideo: https://github.com/scikit-video/scikit-video/issues/154#issuecomment-1445239790
# 修复 skvideo 中 numpy 弃用别名的临时解决方案：https://github.com/scikit-video/scikit-video/issues/154#issuecomment-1445239790
# Will be deleted in MV-2134 when skvideo makes the correction
# 当 skvideo 进行修正时，将在 MV-2134 中删除
from skvideo.io import FFmpegWriter
from enum import Enum


class FlowType(Enum):
    PlaneFitting = "PlaneFitting"
    TimeGradient = "TimeGradient"
    TripletMatching = "TripletMatching"

    def __str__(self):
        return self.value


def parse_args():
    import argparse
    """
    Parse command line arguments.
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description='Metavision Dense Optical Flow sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group(
        "Input", "Arguments related to input sequence.")
    input_group.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="",
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    input_group.add_argument(
        '--replay_factor', type=float, default=1,
        help="Replay Factor. If greater than 1.0 we replay with slow-motion, otherwise this is a speed-up over real-time.")
    input_group.add_argument(
        '--dt-step', type=int, default=33333, dest="dt_step",
        help="Time processing step (in us), used as iteration delta_t period, visualization framerate and accumulation time.")

    algo_settings_group = parser.add_argument_group(
        "Algo Settings", "Arguments related to algorithmic configuration.")
    algo_settings_group.add_argument(
        "--flow-type", dest="flow_type", type=FlowType, choices=list(FlowType),
        default=FlowType.TripletMatching, help="Chosen type of dense flow algorithm to run")
    algo_settings_group.add_argument(
        "-r", "--receptive-field-radius", dest="receptive_field_radius", type=float, default=3,
        help="Radius of the receptive field used for flow estimation, in pixels.")
    algo_settings_group.add_argument("--min-flow", dest="min_flow_mag", type=float,  default=10.,
                                     help="Minimum observable flow magnitude, in px/s.")
    algo_settings_group.add_argument("--max-flow", dest="max_flow_mag", type=float,  default=1000.,
                                     help="Maximum observable flow magnitude, in px/s.")
    algo_settings_group.add_argument("--stc-filter-thr", dest="stc_filter_thr", type=int,  default=40000,
                                     help="Length of the time window for filtering (in us).")
    algo_settings_group.add_argument(
        "--visu-scale", dest="visualization_flow_scale", type=float, default=0.8,
        help="Flow magnitude used to scale the upper bound of the flow visualization, in px/s. If negative, will use 1/5-th of maximum flow magnitude.")

    output_flow_group = parser.add_argument_group(
        "Output flow", "Arguments related to output optical flow.")
    output_flow_group.add_argument(
        "--output-sparse-npy-filename", dest="output_sparse_npy_filename",
        help="If provided, the predictions will be saved as numpy structured array of EventOpticalFlow. In this "
        "format, the flow vx and vy are expressed in pixels per second.")
    output_flow_group.add_argument(
        "--output-dense-h5-filename", dest="output_dense_h5_filename",
        help="If provided, the predictions will be saved as a sequence of dense flow in HDF5 data. The flows are "
        "averaged pixelwise over timeslices of --dt-step. The dense flow is expressed in terms of "
        "pixels per timeslice (of duration dt-step), not in pixels per second.")
    output_flow_group.add_argument(
        '-o', '--out-video', dest='out_video', type=str, default="",
        help="Path to an output AVI file to save the resulting video.")
    output_flow_group.add_argument(
        '--fps', dest='fps', type=int, default=30,
        help="replay fps of output video")
    output_flow_group.add_argument(
        '--no-legend', dest='show_legend', action='store_false',
        help="Set to remove the legend from the display")

    args = parser.parse_args()

    if args.output_sparse_npy_filename:
        assert not os.path.exists(args.output_sparse_npy_filename)
    if args.output_dense_h5_filename:
        assert not os.path.exists(args.output_dense_h5_filename)
    if args.visualization_flow_scale <= 0:
        args.visualization_flow_scale = 1

    return args


def accumulate_estimated_flow_data(
        args, width, height, processing_ts, flow_buffer, all_flow_events, all_dense_flows_start_ts,
        all_dense_flows_end_ts, all_dense_flows):
    """
    Accumulates estimated flow data in buffers to dump estimation results
    将估计的流量数据累积在缓冲区中以转储估计结果
    """
    if args.output_sparse_npy_filename:
        all_flow_events.append(flow_buffer.numpy().copy())
    if args.output_dense_h5_filename:
        all_dense_flows_start_ts.append(
            processing_ts - args.dt_step)
        all_dense_flows_end_ts.append(processing_ts)
        flow_np = flow_buffer.numpy()
        if flow_np.size == 0:
            all_dense_flows.append(
                np.zeros((2, height, width), dtype=np.float32))
        else:
            xs, ys, vx, vy = flow_np["x"], flow_np["y"], flow_np["vx"], flow_np["vy"]
            coords = np.stack((ys, xs))
            abs_coords = np.ravel_multi_index(coords, (height, width))
            counts = np.bincount(abs_coords, weights=np.ones(flow_np.size),
                                 minlength=height*width).reshape(height, width)
            flow_x = np.bincount(
                abs_coords, weights=vx, minlength=height*width).reshape(height, width)
            flow_y = np.bincount(
                abs_coords, weights=vy, minlength=height*width).reshape(height, width)
            mask_multiple_events = counts > 1
            flow_x[mask_multiple_events] /= counts[mask_multiple_events]
            flow_y[mask_multiple_events] /= counts[mask_multiple_events]

            # flow expressed in pixels per delta_t
            flow_x *= args.dt_step * 1e-6
            flow_y *= args.dt_step * 1e-6
            flow = np.stack((flow_x, flow_y)).astype(np.float32)
            all_dense_flows.append(flow)


def dump_estimated_flow_data(
        args, width, height, all_flow_events, all_dense_flows_start_ts, all_dense_flows_end_ts, all_dense_flows):
    """
    Write accumulated flow results to output files
    将累积流量结果写入输出文件
    """
    try:
        if args.output_sparse_npy_filename:
            print("Writing output file: ", args.output_sparse_npy_filename)
            all_flow_events = np.concatenate(all_flow_events)
            np.save(args.output_sparse_npy_filename, all_flow_events)
        if args.output_dense_h5_filename:
            print("Writing output file: ", args.output_dense_h5_filename)
            flow_start_ts = np.array(all_dense_flows_start_ts)
            flow_end_ts = np.array(all_dense_flows_end_ts)
            flows = np.stack(all_dense_flows)
            N = flow_start_ts.size
            assert flow_end_ts.size == N
            assert flows.shape == (N, 2, height, width)
            dirname = os.path.dirname(args.output_dense_h5_filename)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            flow_h5 = h5py.File(args.output_dense_h5_filename, "w")
            flow_h5.create_dataset(
                "flow_start_ts", data=flow_start_ts, compression="gzip")
            flow_h5.create_dataset(
                "flow_end_ts", data=flow_end_ts, compression="gzip")
            flow_h5.create_dataset("flow", data=flows.astype(
                np.float32), compression="gzip")
            flow_h5["flow"].attrs["input_file_name"] = os.path.basename(
                args.event_file_path)
            flow_h5["flow"].attrs["checkpoint_path"] = "metavision_dense_optical_flow"
            flow_h5["flow"].attrs["event_input_height"] = height
            flow_h5["flow"].attrs["event_input_width"] = width
            flow_h5["flow"].attrs["delta_t"] = args.dt_step
            flow_h5.close()
    except Exception as e:
        print(e)
        raise


def main():
    """ Main """
    args = parse_args()

    # Events iterator on Camera or event file
    mv_iterator = EventsIterator(
        input_path=args.event_file_path, delta_t=args.dt_step)

    # Set ERC to 20Mev/s
    if hasattr(mv_iterator.reader, "device") and mv_iterator.reader.device and mv_iterator.reader.device.get_i_erc_module():
        erc_module = mv_iterator.reader.device.get_i_erc_module()
        erc_module.set_cd_event_rate(20000000)
        erc_module.enable(True)

    if args.replay_factor > 0 and not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(
            mv_iterator, replay_factor=args.replay_factor)
    height, width = mv_iterator.get_size()  # Camera Geometry

    # Event Frame Generator
    event_frame_gen = OnDemandFrameGenerationAlgorithm(
        width, height, args.dt_step)

    # Dense Optical Flow Algorithm
    # 密集光流算法 选择
    # The input receptive field radius represents the total area of the neighborhood that is used to estimate flow. We
    # use an algorithm-dependent heuristic to convert this into the search radius to be used for each algorithm.
    # 输入感受野半径表示用于估计流量的邻域总面积。我们使用一种与算法相关的启发式方法将其转换为每个算法使用的搜索半径。
    if args.flow_type == FlowType.PlaneFitting:
        radius = math.floor(args.receptive_field_radius)
        print(
            f"Instantiating PlaneFittingFlowAlgorithm with radius={radius}")
        flow_algo_planefitting = PlaneFittingFlowAlgorithm(
            width, height, radius, -1)
        flow_algo_timegradient = None
        flow_algo_tripletmatching = None
    elif args.flow_type == FlowType.TimeGradient:
        radius = int(args.receptive_field_radius)
        print(
            f"Instantiating TimeGradientFlowAlgorithm with radius={radius}, min_flow={args.min_flow_mag}, bit_cut=2")
        flow_algo_timegradient = TimeGradientFlowAlgorithm(width, height, radius, args.min_flow_mag, 2)
        flow_algo_planefitting = None
        flow_algo_tripletmatching = None
    else:
        radius = 0.5*args.receptive_field_radius
        print(
            f"Instantiating TripletMatchingFlowAlgorithm with radius={radius}, min_flow={args.min_flow_mag}, max_flow={args.max_flow_mag}")
        flow_algo_tripletmatching = TripletMatchingFlowAlgorithm(
            width, height, radius, args.min_flow_mag, args.max_flow_mag)
        flow_algo_planefitting = None
        flow_algo_timegradient = None
    flow_buffer = TripletMatchingFlowAlgorithm.get_empty_output_buffer()

    # Dense Flow Frame Generator
    flow_frame_gen = DenseFlowFrameGeneratorAlgorithm(
        width, height, args.max_flow_mag, args.visualization_flow_scale,
        DenseFlowFrameGeneratorAlgorithm.VisualizationMethod.Arrows,
        DenseFlowFrameGeneratorAlgorithm.AccumulationPolicy.Average)
    flow_legend_img = np.zeros((100, 100, 3), np.uint8)
    flow_frame_gen.generate_legend_image(flow_legend_img)
    legend_mask = flow_legend_img != 0

    # STC filter
    print(
        f"Instantiating SpatioTemporalContrastAlgorithm with thresh={args.stc_filter_thr}")
    stc_filter = SpatioTemporalContrastAlgorithm(
        width, height, args.stc_filter_thr, True)
    events_buf = SpatioTemporalContrastAlgorithm.get_empty_output_buffer()

    all_flow_events = []
    all_dense_flows = []
    all_dense_flows_start_ts = []
    all_dense_flows_end_ts = []

    # Window - Graphical User Interface
    with Window(title="Metavision Dense Optical Flow", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        if args.out_video:
            video_name = args.out_video + ".avi"
            writer = FFmpegWriter(video_name, inputdict={'-r': str(args.fps)}, outputdict={
                '-vcodec': 'libx264',
                '-pix_fmt': 'yuv420p',
                '-r': str(args.fps)
            })

        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        cd_output_img = np.zeros((height, width, 3), np.uint8)
        flow_output_img = np.zeros((height, width, 3), np.uint8)
        combined_output_img = np.zeros((height, width, 3), np.uint8)
        processing_ts = mv_iterator.start_ts

        if args.flow_type == FlowType.PlaneFitting:
            flow_processor = flow_algo_planefitting
        elif args.flow_type == FlowType.TimeGradient:
            flow_processor = flow_algo_timegradient
        else:
            flow_processor = flow_algo_tripletmatching

        # Process events
        for evs in mv_iterator:
            processing_ts += mv_iterator.delta_t

            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()

            # Filter Events using STC
            stc_filter.process_events(evs, events_buf)

            # Generate Frame of Events
            event_frame_gen.process_events(events_buf)
            event_frame_gen.generate(processing_ts, cd_output_img)

            # Estimate the flow events 根据选择的光流算法估算光流
            flow_processor.process_events(events_buf, flow_buffer)

            # 将估算的光流存储在缓存区
            accumulate_estimated_flow_data(
                args, width, height, processing_ts, flow_buffer, all_flow_events, all_dense_flows_start_ts,
                all_dense_flows_end_ts, all_dense_flows)

            # Draw the flow events on top of the events  光流帧生成
            flow_frame_gen.process_events(flow_buffer)
            flow_frame_gen.generate(flow_output_img)

            # Update the display
            cv2.addWeighted(cd_output_img, 0.4, flow_output_img, 0.6, 0, combined_output_img)
            if args.show_legend:
                lh, lw = flow_legend_img.shape[0], flow_legend_img.shape[1]
                combined_output_img[height - 1 - lh: height - 1, 0: lw][legend_mask != 0] = flow_legend_img[legend_mask != 0]
            window.show(combined_output_img)

            flow_output_img.fill(0)

            if args.out_video:
                writer.writeFrame(combined_output_img.astype(np.uint8)[..., ::-1])

            if window.should_close():
                break

    if args.out_video:
        writer.close()
    dump_estimated_flow_data(args, width, height, all_flow_events,
                             all_dense_flows_start_ts, all_dense_flows_end_ts, all_dense_flows)


if __name__ == "__main__":
    main()