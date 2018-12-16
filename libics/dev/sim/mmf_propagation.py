import numpy as np
from scipy import constants

from libics.data import arraydata
from libics.display import plot, plotdefault
from libics.dev.sim.mmf_modes import COORD, RoundStepIndexFiber
from libics.dev.sim.mmf_source import GaussianBeam


###############################################################################


if __name__ == "__main__":

    # ++++ Test data ++++++++

    # Fiber
    numerical_aperture = 0.22
    clad_refr = 1.45
    core_refr = np.sqrt(clad_refr**2 + numerical_aperture**2)
    core_radius = 10e-6
    fiber_length = 200.0
    fiber_type = "round_step"

    # Light source
    center_freq = constants.speed_of_light / 780e-9
    fwhm_freq = 8e-9 * center_freq**2 / constants.speed_of_light
    spec_num = 4000
    beam_rotation = [1e-1, 0, 0]
    beam_offset = [0, 0, 0]
    beam_waist = 10e-6

    # Calculation
    overlap_algorithm = "simpson"
    result = "output"   # "output", "correlation"

    # Plotting
    plot_num = max(50, round(2 * core_radius / constants.c * center_freq))

    # ++++ Calculation ++++++++

    # Light source
    opt_freqs = np.linspace(
        center_freq - 10 * fwhm_freq, center_freq + 10 * fwhm_freq,
        num=spec_num
    )
    spectrum = arraydata.ArrayData()
    spectrum.add_dim(
        offset=opt_freqs[0],
        scale=((opt_freqs[-1] - opt_freqs[0]) / (len(opt_freqs) - 1)),
        name="frequency", symbol=r"\nu", unit="Hz"
    )
    spectrum.add_dim(name="spectral density", symbol="S_A", unit="1/nm")
    spectrum.data = np.exp(-2 * (opt_freqs - center_freq)**2 / fwhm_freq**2)
    gaussian_beam = GaussianBeam(
        spectrum=spectrum, waist=beam_waist,
        rotation=beam_rotation, offset=beam_offset
    )

    # Fiber
    fiber = RoundStepIndexFiber(
        opt_freq=gaussian_beam.center_frequency,
        core_radius=core_radius, core_refr=core_refr, clad_refr=clad_refr
    )
    fiber.calc_modes()
    overlap = fiber.calc_overlap(gaussian_beam, algorithm=overlap_algorithm)

    # ++++ Plotting ++++++++

    # Output
    rmax = 1.1 * core_radius
    rscale = 2 * rmax / (plot_num - 1)
    output = arraydata.ArrayData()
    output.add_dim(offset=-rmax, scale=rscale,
                   name="position", symbol="x", unit="m")
    output.add_dim(offset=-rmax/5, scale=rscale,
                   name="position", symbol="y", unit="m")
    output.add_dim(name="field", symbol="A", unit="arb.")
    x = np.linspace(-rmax, rmax, num=plot_num)
    y = np.linspace(-rmax / 5, rmax / 5, num=plot_num / 5)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    output.data = abs(fiber.output(
        xx, yy, overlap, fiber_length, coord=COORD.CARTESIAN
    ))

    # Plot
    vmax = max(output.data.max(), -output.data.min())
    pcfg = plotdefault.get_plotcfg_arraydata_2d(
        aspect=1, color="RdBu", min=-vmax, max=vmax
    )
    fcfg = plotdefault.get_figurecfg()
    fig = plot.Figure(fcfg, pcfg, data=output)
    fig.plot()
    fig.show()
