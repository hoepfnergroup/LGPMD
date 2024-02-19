def rdf2sq(r, rdf, Qmin, Qmax, ρ):
    import numpy as np
    q = np.linspace(Qmin, Qmax, 1000)
    dr = r[1] - r[0] # Stepsize
    sq   = np.zeros(len(q))
    for j in range (len(q)):
        sq[j] = (1 + 4*np.pi*ρ*np.trapz(r*(rdf-1)*np.sin(q[j]*r),dx = dr)/q[j])
    return q, sq


def compute_simple_rdf(gsd_fname, rmin, rmax, bins):
    import freud
    import gsd.hoomd
    # Reads in traj and creates box from it
    traj = gsd.hoomd.open(gsd_fname, 'r')
    box = freud.box.Box.from_box(traj[0].configuration.box[:3])

    # Establish RDF object to continually add frames into
    rdf = freud.density.RDF(bins=bins, r_min = rmin, r_max=rmax)
    for frame in traj:
        rdf.compute(system=(box,frame.particles.position), reset=False)

    return rdf.bin_centers, rdf.rdf