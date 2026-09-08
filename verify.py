from pathlib import Path
import py_compile
root = Path(__file__).resolve().parent
for p in [
    root/"run_presenter.py",
    root/"presenter"/"server.py",
    root/"simulation"/"world.py",
    root/"simulation"/"controller.py",
    root/"occupancy"/"occupancy_grid.py",
    root/"prediction"/"future_occupancy.py",
    root/"planning"/"trajectory_generator.py",
    root/"planning"/"collision_checker.py",
    root/"planning"/"cost_function.py",
    root/"planning"/"adaptive_planner.py",
]:
    py_compile.compile(str(p), doraise=True)
    print("OK:", p.relative_to(root))
print("All core files compile successfully.")
