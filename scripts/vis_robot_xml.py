import mujoco
import mujoco.viewer
import time
import sys

def main(xml_path):
    # 加载模型
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as e:
        print(f"❌ Failed to load XML file: {e}")
        sys.exit(1)

    # create simulation data
    data = mujoco.MjData(model)

    print(f"✅ Successfully loaded model: {xml_path}")

    # launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # you can add control logic here, like data.ctrl[:] = ...
            mujoco.mj_step(model, data)

            viewer.sync()

            # control simulation frequency (simulate real time)
            time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
    else:
        # Default test files
        xml_files = [
            # "assets/kuavo_s45/xml/biped_s45_collision.xml",
            "/home/ANT.AMAZON.COM/yanjieze/projects/GMR/assets/kuavo_s45/biped_s45_collision.xml"
            # "/home/ANT.AMAZON.COM/yanjieze/projects/GMR/assets/fourier_n1/n1_mocap.xml"
        ]
        
        print("Testing both hand XML files...")
        for xml_file in xml_files:
            print(f"\nTesting: {xml_file}")
            try:
                import mujoco
                model = mujoco.MjModel.from_xml_path(xml_file)
                print(f"✅ {xml_file} loads successfully!")
            except Exception as e:
                print(f"❌ {xml_file} failed: {e}")
        
        # Launch viewer with left hand by default
        xml_file = xml_files[0]
        print(f"\nLaunching viewer with: {xml_file}")
    
    main(xml_file)
