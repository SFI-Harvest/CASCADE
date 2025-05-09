import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random

class CascadePlotting:

    cmap = {
        "y_transformed": "jet",
        "y": "jet",
        "Psi_y": "jet",
    }

    vlim = {
        "y_transformed": (-2, 2),
        "y": (0, 50),
        "Psi_y": (0, 2),
    }

    def test_inverse_functions(cascade):
        tok = cascade.timing.start_time("test_inverse_functions", class_name="CascadePlotting")

        # Test the inverse functions
        S = None
        T = None
        n = 1000
        x = np.linspace(0, 10, n)
        y1 = (np.sin(x) + 1) * 50
        data_dict = {"y": y1}
        y1_transformed = cascade.data_transformation_y(data_dict)
        y1_transformed_back = cascade.data_transformation_y_inv(y1_transformed)

        sinmod_1 = (np.cos(x) + 1.0001) * 50
        sinmod_1_transformed = cascade.data_transformation_sinmod(S, T, sinmod_1)
        sinmod_1_transformed_back = cascade.data_transformation_sinmod_inv(S, T, sinmod_1_transformed)

        # Plot the results
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].plot(x, y1, label="y1", c="red")
        ax[0, 0].plot(x, y1_transformed, label="y1 transformed")
        ax[0, 0].plot(x, y1_transformed_back, label="y1 transformed back", linestyle="--", c="green")
        ax[0, 0].set_title("y1")
        ax[0, 0].legend()
        ax[0,0].set_ylim(np.min(y1), np.max(y1))

        ax[1, 0].plot(x, y1, label="y1",c="red")
        ax[1, 0].plot(x, y1_transformed, label="y1 transformed")
        ax[1, 0].plot(x, y1_transformed_back, label="y1 transformed back", linestyle="--", c="green")
        ax[1, 0].set_title("y1 transformed")
        ax[1, 0].legend()
        ax[1, 0].set_ylim(np.min(y1_transformed), np.max(y1_transformed))
        ax[0, 1].plot(x, sinmod_1, label="sinmod_1",c="red")
        ax[0, 1].plot(x, sinmod_1_transformed, label="sinmod_1 transformed")
        ax[0, 1].plot(x, sinmod_1_transformed_back, label="sinmod_1 transformed back", linestyle="--", c="green")
        ax[0, 1].set_title("sinmod_1")
        ax[0, 1].legend()
        ax[1, 1].plot(x, sinmod_1, label="sinmod_1",c="red")
        ax[1, 1].plot(x, sinmod_1_transformed, label="sinmod_1 transformed")
        ax[1, 1].plot(x, sinmod_1_transformed_back, label="sinmod_1 transformed back", linestyle="--", c="green")
        ax[1, 1].set_title("sinmod_1 transformed")
        ax[1, 1].legend()
        ax[1, 1].set_ylim(np.min(sinmod_1_transformed), np.max(sinmod_1_transformed))
        plt.tight_layout()
        plt.savefig("figures/tests/Cascade/inverse_functions.png")
        plt.close(fig)

        cascade.timing.end_time_id(tok)



    def plot_grid(cascade):

        t = cascade.timing.start_time("plot_grid", class_name="CascadePlotting")

        grid = cascade.grid.grid
        x = grid[:, 0]
        y = grid[:, 1]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(x, y, c="blue", label="Grid points")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Grid points")
        ax.legend()
        plt.tight_layout()
        plt.savefig("figures/tests/Cascade/grid.png")
        plt.close()
        cascade.timing.end_time_id(t)



    def line_data_assignmen(cascade):

        tok = cascade.timing.start_time("line_data_assignment", class_name="CascadePlotting")

        grid_x_ax = cascade.grid.get_x_ax()
        grid_y_ax = cascade.grid.get_y_ax()
        grid_points = cascade.grid.grid
        x_min, x_max = np.min(grid_x_ax), np.max(grid_x_ax)
        y_min, y_max = np.min(grid_y_ax), np.max(grid_y_ax)
        x_start, x_end = np.random.uniform(x_min, x_max), np.random.uniform(x_min, x_max)
        y_start, y_end = np.random.uniform(y_min, y_max), np.random.uniform(y_min, y_max)
        n = 1000
        x = np.linspace(x_start, x_end, n)
        y = np.linspace(y_start, y_end, n)
        S = np.array([x, y]).T
        T = np.linspace(-10, 60*60*3+10, n)
        data_dict = {}
        data_dict["Sx"] = x
        data_dict["Sy"] = y
        data_dict["S"] = S
        data_dict["T"] = T
        y_obs_cor, y_obs , *other_values = cascade.get_prior(S, T) 
        data_dict["y"] = np.clip(y_obs + np.random.normal(0, 0.05, n), 0, None  )
        data_dict["upper_bound"] = np.ones(n)
        data_dict["lower_bound"] = np.zeros(n)
        data_dict["data_source"] = "AUV thor"
        cascade.add_data_to_model(data_dict)
        closest_grid_points = cascade.data_dict["all_data"]["S_grid"]

        plt.figure(figsize=(10, 10))
        plt.scatter(x, y, c="blue", label="Line data points")
        plt.scatter(grid_points[:, 0], grid_points[:, 1], c="red", label="Grid points")
        for i in range(len(closest_grid_points)):
            plt.plot([x[i], closest_grid_points[i, 0]], [y[i], closest_grid_points[i, 1]], c="green", linestyle="--")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Line data assignment")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/tests/Cascade/line_data_assignment.png")
        plt.close()

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        CascadePlotting.plot_path_2D(cascade, ax[0, 0], label="Path", color="blue")
        CascadePlotting.plot_m_y(cascade, ax[0, 1], color="blue")
        CascadePlotting.plot_y_transformed(cascade, ax[0, 1],color="red")
        CascadePlotting.plot_prior_y_corrected(cascade, ax[0, 1], color="red")

        for axx in ax.flatten():
            axx.legend()


        fig.savefig("figures/tests/Cascade/line_data_assignment_fitted.png")
        plt.close(fig)


        S_grid_inds = cascade.get_data_S_grid_inds()
        T_grid_inds = cascade.get_data_T_grid_inds()
        S_T_grid_inds = cascade.get_data_S_T_grid_inds()
        print(f"len line: {np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2):.1f} m")
        #print(f"Unique grid time indices: {np.unique(T_grid_inds)}")
        #print(f"Unique grid space indices: {np.unique(S_grid_inds)}")
        #print(f"Unique grid space-time indices: {np.unique(S_T_grid_inds)}")
        print(f"Number of unique time indices: {len(np.unique(T_grid_inds))}")
        print(f"Number of unique space indices: {len(np.unique(S_grid_inds))}")
        print(f"Number of unique space-time indices: {len(np.unique(S_T_grid_inds))}")


        cascade.print_data_shape()

        cascade.timing.end_time_id(tok)


    def multiple_line_assignments(cascade):

        tok = cascade.timing.start_time("multiple_line_assignments", class_name="CascadePlotting")

        grid_x_ax = cascade.grid.get_x_ax()
        grid_y_ax = cascade.grid.get_y_ax()
        grid_points = cascade.grid.grid
        x_min, x_max = np.min(grid_x_ax), np.max(grid_x_ax)
        y_min, y_max = np.min(grid_y_ax), np.max(grid_y_ax)
        n_lines = 5
        n_points_per_line = 200
        x_start, y_start = np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)
        S_list = []
        T_list = []
        t_start = -10
        for i in range(n_lines):
            x_end, y_end = np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)
            Sx = np.linspace(x_start, x_end, n_points_per_line)
            Sy = np.linspace(y_start, y_end, n_points_per_line)
            S = np.array([Sx, Sy]).T
            T = np.linspace(t_start, t_start + 60 * 60, n_points_per_line)

            S_list.append(S)
            T_list.append(T)

            # Make the data dict
            data_dict = {}
            data_dict["Sx"] = Sx
            data_dict["Sy"] = Sy
            data_dict["S"] = S
            data_dict["T"] = T
            y_obs_cor, y_obs , *other_values = cascade.get_prior(S, T) 
            data_dict["y"] = np.clip(y_obs + np.random.normal(0, 0.05, n_points_per_line), 0, None  )
            data_dict["upper_bound"] = np.ones(n_points_per_line) * 10 + np.random.normal(0, 0.3, n_points_per_line)
            data_dict["lower_bound"] = np.ones(n_points_per_line) * 30 + np.random.normal(0, 0.3, n_points_per_line)
            data_dict["data_source"] = random.choice(["AUV thor", "ASV greta", "AUV roald"])
            cascade.add_data_to_model(data_dict)
            x_start, y_start = x_end, y_end
            t_start += 60 * 60


        # Print the data shapes 
        cascade.print_data_shape()


        # Plot the data
        plt.figure(figsize=(10, 10))
        for i in range(n_lines):
            S = S_list[i]
            plt.scatter(S[:, 0], S[:, 1], c="blue", label="Line data points")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig("figures/tests/Cascade/multiple_line_assignments.png")
        plt.close()

        # Plot the fitted data
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        # Plot the path witch bathc color
        S_all = cascade.get_data_S()
        T_all = cascade.get_data_T()
        batch = cascade.get_data_batch()

        ax[0, 0].scatter(S_all[:, 0], S_all[:, 1], c=batch, label="Line data points, color by batch")
        ax[0, 0].set_xlim(x_min, x_max)
        ax[0, 0].set_ylim(y_min, y_max)
        ax[0, 0].set_title("Line data points, color by batch")

        # 
        m_y = cascade.get_data_m_y()
        mu_y = cascade.get_data_prior_y()
        y = cascade.get_data_y(gridded=True, corrected=True)

        ax[0,1].plot(m_y, label="m_y")
        ax[0,1].plot(mu_y, label="mu_y")
        ax[0,1].scatter(np.arange(len(m_y)), y, c="red", label="y")
        ax[0,1].legend()
        ax[0,1].set_title("m_y and mu_y")

        # Get the S T grid indices
        S_T_grid_inds = cascade.get_data_S_T_grid_inds()
        ax[1, 0].scatter(S_all[:, 0], S_all[:, 1], c=S_T_grid_inds,cmap="jet", label="Line data points, color by batch")
        ax[1, 0].set_xlim(x_min, x_max)
        ax[1, 0].set_ylim(y_min, y_max)
        ax[1, 0].set_title(f"Line data points, color by grid index \n {len(np.unique(S_T_grid_inds))} unique grid indices")

        y_gridded = cascade.get_data_y(gridded=True, corrected=True)
        y_ungridded = cascade.get_data_y(gridded=False, corrected=True)
        ax[1,1].hist(y_ungridded, bins=100, color="blue", alpha=0.5, label="y ungridded", density=True)
        ax[1,1].hist(y, bins=100, color="red", alpha=0.5, label="y gridded", density=True)
        # Set log scale
        ax[1,1].set_yscale("log")
        sd_y = np.std(y)
        sd_y_ungridded = np.std(y_ungridded)
        ax[1,1].set_title(f"y ungridded and gridded \n sd_y: {sd_y:.2f} \n sd_y_ungridded: {sd_y_ungridded:.2f}") 
        ax[1,1].legend()
        plt.tight_layout()


        y_gr = cascade.get_data_y(gridded=True, corrected=False)
        y_ungr = cascade.get_data_y(gridded=False, corrected=False)
        print(f"sd(y_gridded): {np.std(y_gridded):.2f}")
        print(f"sd(y_ungridded): {np.std(y_ungridded):.2f}")
        print(f"sd(y): {np.std(y_gr):.2f}")
        print(f"sd(y ungridded): {np.std(y_ungr):.2f}")
        
        plt.savefig("figures/tests/Cascade/multiple_line_assignments_fitted.png")
        plt.close()

        cascade.timing.end_time_id(tok) 




    def plot_predicions(cascade):

        tok = cascade.timing.start_time("plot_predictions", class_name="CascadePlotting")
        cascade.clean_model()
    
        grid_x_ax = cascade.grid.get_x_ax()
        grid_y_ax = cascade.grid.get_y_ax()
        grid_points = cascade.grid.grid
        x_min, x_max = np.min(grid_x_ax), np.max(grid_x_ax)
        y_min, y_max = np.min(grid_y_ax), np.max(grid_y_ax)


                

        x_start, x_end = np.random.uniform(x_min, x_max), np.random.uniform(x_min, x_max)
        y_start, y_end = np.random.uniform(y_min, y_max), np.random.uniform(y_min, y_max)
        n = 1000
        m = np.random.randint(450, 550)
        Sx = np.linspace(x_start, x_end, n)
        Sy = np.linspace(y_start, y_end, n)
        S = np.array([Sx, Sy]).T
        
        T = np.linspace(-10, 60*60*1+10, n)
        S_pred = S[m:] 
        T_pred = T[m:]
        S = S[:m]
        T = T[:m]
        print(T.shape, S.shape, S_pred.shape, T_pred.shape)
        data_dict = {}
        data_dict["Sx"] = Sx[:m]
        data_dict["Sy"] = Sy[:m]
        data_dict["S"] = S
        data_dict["T"] = T
        y_obs_cor, y_obs , *other_values = cascade.get_prior(S, T)
        data_dict["y"] = np.clip(y_obs + np.random.normal(0, 0.05, m), 0, None  )
        data_dict["upper_bound"] = np.ones(m) * 10
        data_dict["lower_bound"] = np.ones(m) * 10
        data_dict["data_source"] = random.choice(["AUV thor", "ASV greta", "AUV roald"])
        cascade.add_data_to_model(data_dict)
        

        y_true_cor, y_true, *_ = cascade.get_prior(S_pred, T_pred)
        predictions = cascade.predict(S_pred, T_pred)

        Psi = predictions["Psi_y"]
        dPsi = np.diag(Psi)
        m_y = predictions["m_y"]


        # Plot the data
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].scatter(S[:, 0], S[:, 1], c="blue", label="Data in model")
        ax[0, 0].scatter(S_pred[:, 0], S_pred[:, 1], c="red", label="Predictions")
        ax[0, 0].set_xlim(x_min, x_max)
        ax[0, 0].set_ylim(y_min, y_max)
        ax[0, 0].set_title("Data in model and predictions")
        ax[0, 0].legend()

        # Load data from the model
        ax[0,1].plot(m_y, label="m_y")
        y_true_transformed = cascade.data_transformation_y({"y":y_true})
        ax[0,1].plot(y_true_transformed, label="y_true")
        ax[0,1].set_ylim(min(m_y), max(m_y))
        ax[0,1].fill_between(np.arange(len(m_y)), m_y - 1.96 * np.std(dPsi), m_y + 1.96 * np.std(dPsi), alpha=0.2, label="95% CI")

        plt.savefig("figures/tests/Cascade/predictions.png")
        plt.close(fig)



    def plot_predict_field(cascade):

        tok = cascade.timing.start_time("predict_field", class_name="CascadePlotting")
    
        grid_x_ax = cascade.grid.get_x_ax()
        grid_y_ax = cascade.grid.get_y_ax()
        grid_points = cascade.grid.grid
        x_min, x_max = np.min(grid_x_ax), np.max(grid_x_ax)
        y_min, y_max = np.min(grid_y_ax), np.max(grid_y_ax)

        t = cascade.get_data_T()[-1]

        field_pred, S_field, T_field = cascade.predict_field(t=t)
        field_pred_t_b = cascade.data_transformation_y_inv(field_pred["m_y"])
        _, y_true, *_ = cascade.get_prior(S_field, T_field)
        y_true_transformed = cascade.data_transformation_sinmod(S_field, T_field, y_true)
        y_transformed_back = cascade.data_transformation_y_inv(field_pred["m_y"])

        S_path = cascade.get_data_S()

        fig, ax = plt.subplots(2,3, figsize=(20, 20))
        # Every other column is a colorbar 


        #### Plot the predicions
        ax[0,0].scatter(S_field[:, 0], S_field[:, 1], c=field_pred["m_y"],
                        cmap="jet", label="m_y predictions",
                        vmin=CascadePlotting.vlim["y_transformed"][0], vmax=CascadePlotting.vlim["y_transformed"][1])
        CascadePlotting.add_cbar(ax[0, 0], fig, type="y_transformed", label="m_y predictions")

        ax[1,0].scatter(S_field[:, 0], S_field[:, 1], c=field_pred_t_b,
                        cmap="jet", label="m_y transformed back predictions",
                        vmin=CascadePlotting.vlim["y"][0], vmax=CascadePlotting.vlim["y"][1])
        CascadePlotting.add_cbar(ax[1, 0], fig, type="y", label="m_y transformed back predictions")


        #### Plot the true field
        ax[0,1].scatter(S_field[:, 0], S_field[:, 1], c=y_true_transformed,
                        cmap="jet", label="True field",
                        vmin=CascadePlotting.vlim["y_transformed"][0], vmax=CascadePlotting.vlim["y_transformed"][1])
        CascadePlotting.add_cbar(ax[0, 1], fig, type="y_transformed", label="True field")
        ax[1,1].scatter(S_field[:, 0], S_field[:, 1], c=y_true,
                        cmap="jet", label="True field",
                        vmin=CascadePlotting.vlim["y"][0], vmax=CascadePlotting.vlim["y"][1])
        CascadePlotting.add_cbar(ax[1, 1], fig, type="y", label="True field")

        #### Plot the uncertainty
        ax[0,2].scatter(S_field[:, 0], S_field[:, 1], c=field_pred["dPsi_y"],
                        cmap="jet", label="Uncertainty",
                        vmin=CascadePlotting.vlim["Psi_y"][0], vmax=CascadePlotting.vlim["Psi_y"][1])
        CascadePlotting.add_cbar(ax[0, 2], fig, type="Psi_y", label="Uncertainty")
        CascadePlotting.plot_path_2D(cascade, ax[1, 2], label="Path", color="blue")
        CascadePlotting.add_cbar(ax[1, 2], fig, type="Psi_y", label="Dummy")

        for axx in ax.flatten():
            axx.set_xlim(x_min, x_max)
            axx.set_ylim(y_min, y_max)
            axx.set_xlabel("x")
            axx.set_ylabel("y")
            axx.legend()
        plt.tight_layout()
        plt.savefig("figures/tests/Cascade/predict_field.png")
        plt.close(fig)


        cascade.timing.end_time_id(tok)


    def add_cbar(ax, fig, type="y", label="", **kwargs):
        """
        Add colorbar to the plot.
        """
        cmap = kwargs.get("cmap", CascadePlotting.cmap[type])
        vlim = kwargs.get("vlim", CascadePlotting.vlim[type])
        norm = mpl.colors.Normalize(vmin=vlim[0], vmax=vlim[1])
        cbo = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="vertical")
    


    def plot_path_2D(cascade, ax, **kwargs):
        """
        Plot the path in 2D
        """
        S = cascade.get_data_S()
        label = kwargs.get("label", "Path")
        color = kwargs.get("color", "blue")
        ax.plot(S[:, 0], S[:, 1], c=color, label=label)


    def plot_T_y_transformed(cascade, ax, **kwargs):
        """
        Plot the T and y_transformed
        """
        T = cascade.get_data_T(gridded=True)
        y_transformed = cascade.get_data_y(gridded=True, corrected=True)
        label = kwargs.get("label", "T and y_transformed")
        color = kwargs.get("color", "red")
        ax.scatter(T, y_transformed, c=color, label=label, alpha=0.5, marker="x")

    def plot_y_transformed(cascade, ax, **kwargs):
        y_transformed = cascade.get_data_y(gridded=True, corrected=True)
        label = kwargs.get("label", "y_transformed")
        color = kwargs.get("color", "red")
        x = np.arange(len(y_transformed)) + kwargs.get("offset", 0)
        ax.scatter(x, y_transformed, c=color, label=label)


    def plot_T_my(cascade, ax,ci=True, **kwargs):
        """
        Plot the T and m_y
        """
        T = cascade.get_data_T(gridded=True)
        m_y = cascade.get_data_m_y()
        label = kwargs.get("label", "T and m_y")
        color = kwargs.get("color", "blue")
        ax.plot(T, m_y, c=color, label=label)

        if ci:
            dPsi = np.diag(cascade.get_data_Psi_y())
            ax.fill_between(T, m_y - 1.96 * np.sqrt(dPsi), m_y + 1.96 * np.sqrt(dPsi), alpha=0.2, label="95% CI m_y", color=color)

    def plot_m_y(cascade, ax, ci=True, **kwargs):
        """
        plot the m_y vs the distance to the path
        """
        m_y = cascade.get_data_m_y()
        label = kwargs.get("label", "m_y")
        color = kwargs.get("color", "blue")
        x = np.arange(len(m_y)) + kwargs.get("offset", 0)
        ax.plot(x, m_y, c=color, label=label)
        if ci:
            dPsi = np.diag(cascade.get_data_Psi_y())
            ax.fill_between(x, m_y - 1.96 * np.sqrt(dPsi), m_y + 1.96 * np.sqrt(dPsi), alpha=0.2, label="95% CI m_y", color=color)

    def plot_prior_y_corrected(cascade, ax, **kwargs):
        """
        Plot the prior corrected data
        """
        prior_corrected = cascade.get_data_prior_y(gridded=True, corrected=True)
        label = kwargs.get("label", "Prior corrected")
        color = kwargs.get("color", "red")
        x = np.arange(len(prior_corrected)) + kwargs.get("offset", 0)
        ax.plot(x, prior_corrected, c=color, label=label)




        