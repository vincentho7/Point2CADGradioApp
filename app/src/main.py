import gradio as gr
import os
import argparse
import multiprocessing
import numpy as np
import os
import torch
import trimesh
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

from point2cad.fitting_one_surface import process_one_surface
from point2cad.io_utils import save_unclipped_meshes, save_clipped_meshes, save_topology
from point2cad.utils import seed_everything, continuous_labels, normalize_points, make_colormap_optimal
from point2cad.main import process_multiprocessing, process_one_surface, process_singleprocessing

class ModelConfig():
    def __init__(self, path_in, path_out, validate_checkpoint_path = None, silent=True, seed=42, max_parallel_surfaces=4, 
                num_inr_fit_attempts=1, surfaces_multiprocessing=1):
        self.path_in = path_in
        self.path_out = path_out
        self.validate_checkpoint_path = validate_checkpoint_path
        self.silent = silent
        self.seed = seed
        self.max_parallel_surfaces = max_parallel_surfaces
        self.num_inr_fit_attempts = num_inr_fit_attempts
        self.surfaces_multiprocessing = surfaces_multiprocessing
            
def xyz_to_obj(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        for line in lines:
            data = line.strip().split()
            if len(data) == 3:  # .xyz format 
                x, y, z = data
                f.write(f"v {x} {y} {z}\n")
            elif len(data) == 6:  # .xyzc format 
                x, y, z, r, g, b = data
                f.write(f"v {x} {y} {z} {r} {g} {b}\n")
            else:
                print("Invalid format. Skipping line.")
    return output_file

def ply_to_obj(input_file):
    mesh = trimesh.load(input_file)
    output_file = f"{input_file.rsplit('.', 1)[0]}.obj"
    mesh.export(output_file, overwrite=True)
    return output_file

if __name__ == "__main__":
    #device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    color_list = make_colormap_optimal()
    folder_path = "./assets"
    path_out = "./out"
    os.makedirs(path_out, exist_ok=True)
    os.makedirs("{}/unclipped".format(path_out), exist_ok=True)
    os.makedirs("{}/clipped".format(path_out), exist_ok=True)
    os.makedirs("{}/topo".format(path_out), exist_ok=True)

    files_in_folder = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    def process_data(path_in, path_out, validate_checkpoint_path, silent, seed, max_parallel_surfaces, 
                 num_inr_fit_attempts, surfaces_multiprocessing ):
        
        full_path = '{folder_path}/{path_in}'
        gr.Info("Starting process")
        if name is None:
            gr.Warning("Name is empty")
        ...
        if success == False:
            raise gr.Error("Process failed")

        info = [full_path, path_out, validate_checkpoint_path, silent, seed, max_parallel_surfaces, \
                 num_inr_fit_attempts, surfaces_multiprocessing]
        cfg = ModelConfig(full_path, path_out, validate_checkpoint_path, silent, seed, max_parallel_surfaces, 
                num_inr_fit_attempts, surfaces_multiprocessing)
        
    #     seed_everything(cfg.seed)

    #     fn_process = process_singleprocessing
    #     if surfaces_multiprocessing:
    #         multiprocessing.set_start_method("spawn", force=True)
    #         fn_process = process_multiprocessing

    # # ============================ load points ============================
    #     points_labels = np.loadtxt(cfg.path_in).astype(np.float32)
    #     assert (
    #         points_labels.shape[1] == 4
    #     ), "This pipeline expects annotated point clouds (4 values per point). Refer to README for further instructions"
    #     points = points_labels[:, :3]
    #     labels = points_labels[:, 3].astype(np.int32)
    #     labels = continuous_labels(labels)

    #     points = normalize_points(points)
    #     if device.type == "cuda":
    #         torch.cuda.empty_cache()

    #     uniq_labels = np.unique(labels)

    #     out_meshes = fn_process(cfg, uniq_labels, points, labels, device)

    #     # ============================ save unclipped meshes ============================
    #     print("Saving unclipped meshes...")
    #     pm_meshes = save_unclipped_meshes(
    #         out_meshes, color_list, "{}/unclipped/mesh.ply".format(path_out)
    #     )

    #     # ============================ save clipped meshes ==============================
    #     print("Saving clipped meshes...")
    #     clipped_meshes = save_clipped_meshes(
    #         pm_meshes, out_meshes, color_list, "{}/clipped/mesh.ply".format(path_out)
    #     )

    #     # ============================ get edges and corners ============================
    #     print("Saving topology (edges and corners)...")
    #     save_topology(clipped_meshes, "{}/topo/topo.json".format(path_out))
    #     mod_path = xyz_to_obj(path_in, f"{path_in.rsplit('.', 1)[0]}.obj")
    #     uncl_path = ply_to_obj("{}/unclipped/mesh.ply".format(path_out)) 
    #     cli_path = ply_to_obj("{}/clipped/mesh.ply".format(path_out)) 
    #     print("Done")
        info_text = "iNFORMATION".join(map(str, info))
        return info_text #, mod_path, uncl_path, cli_path
        #return mod_path, uncl_path, cli_path

    with gr.Blocks(title="Point2CAD pipeline", theme=gr.themes.Soft()) as demo:
        with gr.Row():
            with gr.Tab("Set Evaluation"):
                file_dropdown = gr.Dropdown(choices=files_in_folder, label="Select a file")
                out_folder = gr.Textbox(path_out, label="output folder", interactive=False)
                validate_checkpoint_path = gr.Textbox(value=None, label="validate_checkpoint_path")
                seed_value = gr.Number(label="Seed", value=42, interactive=True)
                silent_cbox = gr.Checkbox(value=True, label="Silent")
                max_par = gr.Number(label="max_parallel_surfaces", value=4, interactive=True)
                inr_attempts = gr.Number(label="num_inr_fit_attempts", value=1, interactive=True)
                surf_multiproc = gr.Number(label="surfaces_multiprocessing", value=1, interactive=True)
            infobox = gr.Textbox(label="information", value="Hello", interactive=False)
            data = gr.Model3D(label="Point Cloud",  interactive=True)   

        with gr.Row():
            with gr.Tab("Reconstruction Points Cloud to CAD"):
                unclipped = gr.Model3D(label="Unclipped mesh CAD", interactive=True)
                clipped = gr.Model3D(label="Clipped mesh CAD", interactive=True)
        
        process_button = gr.Button("Apply Reconstruction")
        process_button.click(process_data, inputs=[file_dropdown, out_folder, validate_checkpoint_path, silent_cbox, seed_value, max_par, 
                                inr_attempts, surf_multiproc], 
                                outputs=[infobox]) 
#                                outputs=[infobox, data, unclipped, clipped])
        
    demo.queue().launch( height=1500, debug=True)

