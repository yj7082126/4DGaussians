input_folder="/data/3D_dataset/datasets/kubric4d/test"
folders=($(ls -d $input_folder/*/ 2>/dev/null))
port=6111
export CUDA_VISIBLE_DEVICES=6

for folder in "${folders[@]:76:8}"; do
    name="$(basename "$folder")"
    root_dir="$input_folder/$name"
    workdir="samples/$name"

    echo "Processing $workdir"
    # python scripts/kubric2colmap.py $root_dir $workdir

    # colmap feature_extractor --database_path $workdir/colmap/database.db --image_path $workdir/colmap/images \
    # --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 \
    # --SiftExtraction.estimate_affine_shape 1 \
    # --SiftExtraction.domain_size_pooling 1
    # python database.py --database_path $workdir/colmap/database.db --txt_path $workdir/colmap/sparse_custom/cameras.txt

    # colmap exhaustive_matcher \
    # --database_path $workdir/colmap/database.db \
    # --SiftMatching.guided_matching true
    # colmap point_triangulator \
    # --database_path $workdir/colmap/database.db \
    # --image_path $workdir/colmap/images \
    # --input_path $workdir/colmap/sparse_custom \
    # --output_path $workdir/colmap/sparse/0

    # colmap image_undistorter \
    # --image_path $workdir/colmap/images \
    # --input_path $workdir/colmap/sparse/0 \
    # --output_path $workdir/colmap/dense/workspace
    # colmap patch_match_stereo --workspace_path $workdir/colmap/dense/workspace
    # colmap stereo_fusion --workspace_path $workdir/colmap/dense/workspace --output_path $workdir/colmap/dense/workspace/fused.ply

    # python scripts/downsample_point.py $workdir/colmap/dense/workspace/fused.ply $workdir/colmap/points3D_multipleview.ply

    python train.py -s $workdir/colmap --port $port  --model_path $workdir/outputs_v2 --configs arguments/multipleview/scn02716.py --device 'cuda:0'
done

# CUDA_VISIBLE_DEVICES=2, python train.py -s samples/scn02902/colmap --port 6017 --expname samples/scn02902/output --configs arguments/multipleview/scn02716.py --kubric