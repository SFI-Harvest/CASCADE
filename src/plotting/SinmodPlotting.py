import matplotlib.pyplot as plt
import numpy as np

class SinmodPlotting:



    def plot_inside_inds(sinmod, **kwargs):


        land_inds_x, land_inds_y = sinmod.sinmod_data["land_ind"]
        ocean_inds_x, ocean_inds_y = sinmod.sinmod_data["ocean_ind"]
        boundary_inds = sinmod.sinmod_data["inds_inside_boundary"]
        xxc = sinmod.sinmod_data["xxc"]
        yyc = sinmod.sinmod_data["yyc"]
        yc = sinmod.sinmod_data["xc"]
        xc = sinmod.sinmod_data["yc"]


        fig, ax = plt.subplots(1, 3, figsize=(15, 10))

        

        print("land_inds", land_inds_y, land_inds_x)
        print("boundary_inds", boundary_inds)


        ax[0].scatter(xc[land_inds_x], xc[land_inds_y], c="red", label="Land")
        ax[1].scatter(xc[ocean_inds_x], yc[ocean_inds_y], c="blue", label="Ocean")
        ax[2].scatter(xxc[boundary_inds], yyc[boundary_inds], c="green", label="Boundary")

        for axx in ax:
            axx.set_xlim(sinmod.sinmod_data["x_min"], sinmod.sinmod_data["x_max"])
            axx.set_ylim(sinmod.sinmod_data["y_min"], sinmod.sinmod_data["y_max"])
            axx.set_xlabel("x")
            axx.set_ylabel("y")
        plt.tight_layout()
        plt.show()
        fig.savefig("figures/tests/Sinmod/inside_inds.png")


    def test_interpolation(sinmod):

        t_tok = sinmod.timing.start_time("test_interpolation", class_name="SinmodPlotting")

        print("#" * 20)
        print("Testing interpolation")
        print("#" * 20)

       # Testing interpolation
        xx = sinmod.sinmod_data["flatten"]["xc"]
        yy = sinmod.sinmod_data["flatten"]["yc"]
        time_stamps = sinmod.sinmod_data["timestamp_seconds"]
        valid_inds = sinmod.sinmod_data["flatten"]["valid_points"]
        xx = xx[valid_inds]
        yy = yy[valid_inds]
        S = np.array([xx, yy]).T
        calanus = sinmod.sinmod_data["flatten"]["calanus_finmarchicus"]

        fig, ax = plt.subplots(2, 3, figsize=(15, 10))

        print("xx.shape", xx.shape)
        print("yy.shape", yy.shape)
        print("calanus.shape", calanus.shape)
        print("calanus[0,valid_inds].shape", calanus[0,valid_inds].shape)
        print("S.shape", S.shape)
        print("valid_inds.shape", valid_inds.shape)
        
        for i, t in enumerate(time_stamps):
            print("t", t)

            ax[0, i].scatter(xx, yy, c=calanus[i,valid_inds], label="Calanus ture")
            T = time_stamps[i]
            T = np.repeat(T, len(xx)) + np.random.normal(0, 0.1, len(xx))
            interpolate_values = sinmod.get_calanus_interpolate_S_T(S, T)

            #ax[0, i].scatter(xx, yy, c=calanus[i], label="Calanus")
            ax[0, i].set_title(f"Calanus at time {t}")
            ax[1, i].scatter(xx, yy, c=interpolate_values, label="Interpolated")
            ax[1, i].set_title(f"Interpolated at time {t}")

        xmax, xmin = np.max(xx), np.min(xx)
        ymax, ymin = np.max(yy), np.min(yy)
        for axx in ax.flatten():
            axx.set_xlim(xmin, xmax)
            axx.set_ylim(ymin, ymax)
            axx.set_xlabel("x")
            axx.set_ylabel("y")
        plt.tight_layout()
        plt.savefig("figures/tests/Sinmod/interpolation.png")
        plt.close(fig)

        sinmod.timing.end_time_id(t_tok)



    def plot_different_inds(sinmod):

        t = sinmod.timing.start_time("plot_different_inds", class_name="SinmodPlotting")

        land_inds = sinmod.sinmod_data["flatten"]["land_inds"]
        ocean_inds = sinmod.sinmod_data["flatten"]["ocean_inds"]
        boundary_inds = sinmod.sinmod_data["flatten"]["inds_inside_boundary"]
        valid_inds = sinmod.sinmod_data["flatten"]["valid_points"]
        shore_inds = sinmod.sinmod_data["flatten"]["shore_inds"]

        inds_dict = {
            "land": land_inds,
            "ocean": ocean_inds,
            "boundary": boundary_inds,
            "valid_points": valid_inds,
            "shore": shore_inds
        }

        xx, yy = sinmod.sinmod_data["xxc"], sinmod.sinmod_data["yyc"]
        xmax, xmin = np.max(xx), np.min(xx)
        ymax, ymin = np.max(yy), np.min(yy)
        fig, ax = plt.subplots(2,3, figsize=(15, 10))

        for i, axx in enumerate(ax.flatten()):
            if i >= len(inds_dict):
                continue
            key = list(inds_dict.keys())[i]
            print("key", key)
            print("inds_dict[key]", inds_dict[key])
            axx.scatter(xx[inds_dict[key]], yy[inds_dict[key]], label=key)
            axx.set_title(key)
            axx.set_xlim(xmin, xmax)
            axx.set_ylim(ymin, ymax)
            axx.set_xlabel("x")
            axx.set_ylabel("y")
        plt.tight_layout()
        plt.savefig("figures/tests/Sinmod/inds.png")
        plt.close(fig)

        sinmod.timing.end_time_id(t)
