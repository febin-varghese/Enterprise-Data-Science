import numpy as np
from src.data_handling import Data
from src.visualize import Visualize, VisualizeSIR


def calculation(data_obj):
    test_data_reg = np.array([2, 4, 6])
    result = data_obj.get_doubling_time_via_regression(test_data_reg)
    print('the test slope is: ' + str(result))

    data_obj.calc_filtered_data()
    data_obj.calc_doubling_rate()
    data_obj.calc_doubling_rate('confirmed_filtered')
    data_obj.save_final_results()


def data_science_project(pull_data=False, sir_dashboard=True):
    data_object = Data(pull_data)
    if pull_data:  # process data only if new data is pulled
        # process data
        data_object.store_relational_data()
        # filtering and doubling rate calculation
        calculation(data_object)
    if sir_dashboard:
        # SIR Model
        sir_vis_object = VisualizeSIR()
        sir_vis_object.app.run_server(debug=True, use_reloader=False)
    else:
        # visualization
        visualization_object = Visualize(data_object.final_data_path)
        visualization_object.app.run_server(debug=True, use_reloader=False)


if __name__ == "__main__":
    visualize_data = False
    if visualize_data:
        data_science_project(False, False)
    else:
        # visualize SIR model curves
        data_science_project(False, True)
