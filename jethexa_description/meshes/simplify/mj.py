# run_jethexa_fixed.py
import os
import glob
import shutil
import mujoco
from mujoco.viewer import launch_passive

class JethexaSimulator:
    def __init__(self, model_path: str):
        # Ensure mesh directory exists and contains all .stl files
        xml_dir = os.path.dirname(os.path.abspath(model_path))
        meshes_dir = os.path.join(xml_dir, 'meshes')
        if not os.path.isdir(meshes_dir):
            os.makedirs(meshes_dir, exist_ok=True)
            # Copy all STL files into the meshes directory
            for mesh_file in glob.glob(os.path.join(xml_dir, '*.stl')):
                shutil.copy(mesh_file, meshes_dir)

        # Change working dir so that meshdir in XML resolves correctly
        os.chdir(xml_dir)

        # Load model and data
        self.model = mujoco.MjModel.from_xml_path(os.path.basename(model_path))
        self.data = mujoco.MjData(self.model)

        # Initialize passive viewer
        self.viewer = launch_passive(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def render(self):
        self.viewer.render()

    def run(self):
        try:
            while True:
                self.step()
                self.render()
        except KeyboardInterrupt:
            print("Simulation terminated.")

if __name__ == "__main__":
    sim = JethexaSimulator("jethexa.xml")
    sim.run()
