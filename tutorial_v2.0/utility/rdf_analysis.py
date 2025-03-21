def snap_molecule_indices(snap):
    import freud
    system = freud.AABBQuery.from_system(snap)
    num_query_points = num_points = snap.particles.N
    query_point_indices = snap.bonds.group[:, 0]
    point_indices = snap.bonds.group[:, 1]
    distances = system.box.compute_distances(
        system.points[query_point_indices], system.points[point_indices])
    nlist = freud.NeighborList.from_arrays(
        num_query_points, num_points, query_point_indices, point_indices, distances)
    cluster = freud.cluster.Cluster()
    cluster.compute(system=system, neighbors=nlist)
    return cluster.cluster_idx

def intermolecular_rdf(
    gsdfile,
    A_name,
    B_name,
    start=0,
    stop=None,
    r_max=None,
    r_min=0,
    bins=1000,
    exclude_bonded=True,):
    import numpy as np
    import gsd.hoomd
    import freud
    with gsd.hoomd.open(gsdfile) as trajectory:
        snap = trajectory[0]
        if r_max is None:
            # Use a value just less than half the maximum box length.
            r_max = np.nextafter(
                np.max(snap.configuration.box[:3]) * 0.5, 0, dtype=np.float32)
        rdf = freud.density.RDF(bins=bins, r_max=r_max, r_min=r_min)
        type_A = snap.particles.typeid == snap.particles.types.index(A_name)
        type_B = snap.particles.typeid == snap.particles.types.index(B_name)
        if exclude_bonded:
            molecules = snap_molecule_indices(snap)
            molecules_A = molecules[type_A]
            molecules_B = molecules[type_B]
        for snap in trajectory[start:stop]:
            A_pos = snap.particles.position[type_A]
            if A_name == B_name:
                B_pos = A_pos
                exclude_ii = True
            else:
                B_pos = snap.particles.position[type_B]
                exclude_ii = False
            box = snap.configuration.box
            system = (box, A_pos)
            aq = freud.locality.AABBQuery.from_system(system)
            nlist = aq.query(
                B_pos, {"r_max": r_max, "exclude_ii": exclude_ii}
            ).toNeighborList()
            if exclude_bonded:
                pre_filter = len(nlist)
                indices_A = molecules_A[nlist.point_indices]
                indices_B = molecules_B[nlist.query_point_indices]
                nlist.filter(indices_A != indices_B)
                post_filter = len(nlist)
            rdf.compute(aq, neighbors=nlist, reset=False)
        normalization = post_filter / pre_filter if exclude_bonded else 1
    return rdf, normalization
        
def rdf2sq(r, rdf, Qmin, Qmax, ρ):
    import numpy as np
    q = np.linspace(Qmin, Qmax, 1000)
    dr = r[1] - r[0] # Stepsize
    sq   = np.zeros(len(q))
    for j in range (len(q)):
        sq[j] = (1 + 4*np.pi*ρ*np.trapz(r*(rdf-1)*np.sin(q[j]*r),dx = dr)/q[j])
    return q, sq