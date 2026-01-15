from tools import gen_real_point
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        '--mesh_path', default='/root/workspace/ObjectLocation/data/map/combine.obj', type=str, help='Path of map mesh')
    parser.add_argument('--duration', default=60, type=int,
                        help='Required amount of simulated data,unit: second')
    parser.add_argument('--objs_cfg', default="data/points_2.json", type=str,
                        help='Configuration file for target start and end points')
    parser.add_argument('--cameras_cfg', default="data/cameras_2.json", type=str,
                        help='Configuration file for camera parameters')
    parser.add_argument('--obj_attr_output', default="simulated_data/objects.json", type=str,
                        help='The name of the object attribute output file, default is "3d_rotation.mp4"')
    parser.add_argument('--vis_output', default="simulated_data/3d_rotation.mp4", type=str,
                        help='The name of the visual output file, default is "3d_rotation.mp4"')

    args = parser.parse_args()

    print(args.mesh_path)

    gen_real_point(args)
