def mie_fluid(it, m, spacing, name, timestep, n, sigma, epsilon, kbT, density, eq_time, production_time):
    import math
    import numpy as np
    import itertools
    import hoomd
    import gsd.hoomd
    
    # Initial configuration setup
    N_particles = 4 * m**3
    K = math.ceil(N_particles**(1 / 3))
    L = K * spacing * sigma
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))

    # Create the snapshot in GSD for hoomd to read in l8r
    snapshot = gsd.hoomd.Frame()
    snapshot.particles.N = N_particles
    snapshot.particles.position = position[0:N_particles]
    snapshot.particles.typeid = [0] * N_particles
    snapshot.configuration.box = [L, L, L, 0, 0, 0]
    snapshot.particles.types = [name]
    with gsd.hoomd.open(name='lattice' + str(it) +'.gsd', mode='w') as f:
        f.append(snapshot)

    # Pair hoomd to the cpu and create basic sim objects
    cpu = hoomd.device.CPU()
    sim = hoomd.Simulation(device=cpu, seed=1)
    sim.create_state_from_gsd(filename='lattice' + str(it) +'.gsd')
    integrator = hoomd.md.Integrator(dt = timestep)
    cell = hoomd.md.nlist.Cell(buffer=0.4)
    
    # Potential energy function
    mie = hoomd.md.pair.Mie(nlist=cell)
    mie.params[(name, name)] = dict(epsilon=epsilon, sigma=sigma, n=n, m=6)
    mie.r_cut[(name, name)] = 3*sigma
    integrator.forces.append(mie)
    
    # Create thermostat and fix the volume to make an NVT sim
    mttk = hoomd.md.methods.thermostats.MTTK(kT=kbT, tau=100*timestep)
    nvt = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All(),thermostat=mttk)
    integrator.methods.append(nvt) # Binds it to the integrator
    sim.operations.integrator = integrator

    # Thermalize the simulation
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kbT)
    sim.run(5_000)
    
    # Use the input target density to indicate how to rehsape the box
    ramp = hoomd.variant.Ramp(A=0, B=1, t_start=sim.timestep, t_ramp=10000) # Indicates how fast to apply box reszizing
    rho = sim.state.N_particles / sim.state.box.volume
    initial_box = sim.state.box
    final_box = hoomd.Box.from_box(initial_box) 
    final_rho = density
    final_box.volume = sim.state.N_particles / final_rho
    box_resize_trigger = hoomd.trigger.Periodic(10)
    box_resize = hoomd.update.BoxResize(box1=initial_box, box2=final_box, variant=ramp, trigger=box_resize_trigger)
    sim.operations.updaters.append(box_resize)
    sim.run(10_000)

    # Equilibrate Simulation
    sim.run(eq_time)
    sim.operations.updaters.remove(box_resize)
    
    
    gsd_writer = hoomd.write.GSD(filename='mie' + str(it) +'.gsd',
                              trigger=hoomd.trigger.Periodic(1000),
                              mode='wb')
    sim.operations.writers.append(gsd_writer)
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermodynamic_properties)
    logger = hoomd.logging.Logger()
    logger.add(thermodynamic_properties)
    gsd_writer.log = logger
    sim.run(production_time)
    
def table_fluid(it, m, spacing, name, timestep, kbT, density, eq_time, production_time, V, F, rmin, rcut):
    import math
    import numpy as np
    import itertools
    import hoomd
    import gsd.hoomd
    
    # Initial configuration setup
    N_particles = 4 * m**3
    K = math.ceil(N_particles**(1 / 3))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))

    # Create the snapshot in GSD for hoomd to read in l8r
    snapshot = gsd.hoomd.Frame()
    snapshot.particles.N = N_particles
    snapshot.particles.position = position[0:N_particles]
    snapshot.particles.typeid = [0] * N_particles
    snapshot.configuration.box = [L, L, L, 0, 0, 0]
    snapshot.particles.types = [name]
    with gsd.hoomd.open(name='lattice' + str(it) +'.gsd', mode='w') as f:
        f.append(snapshot)

    # Pair hoomd to the cpu and create basic sim objects
    cpu = hoomd.device.CPU()
    sim = hoomd.Simulation(device=cpu, seed=1)
    sim.create_state_from_gsd(filename='lattice' + str(it) +'.gsd')
    integrator = hoomd.md.Integrator(dt = timestep)
    cell = hoomd.md.nlist.Cell(buffer=0.4)
    
    # Potential energy function
    table = hoomd.md.pair.Table(nlist=cell)
    table.params[(name, name)] = dict(r_min=rmin, U=V, F=F)
    table.r_cut[(name, name)] = rcut
    integrator.forces.append(table)
    
    # Create thermostat and fix the volume to make an NVT sim
    mttk = hoomd.md.methods.thermostats.MTTK(kT=kbT, tau=100*timestep)
    nvt = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All(),thermostat=mttk)
    integrator.methods.append(nvt) # Binds it to the integrator
    sim.operations.integrator = integrator

    # Thermalize the simulation
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kbT)
    sim.run(5_000)
    
    # Use the input target density to indicate how to rehsape the box
    ramp = hoomd.variant.Ramp(A=0, B=1, t_start=sim.timestep, t_ramp=10000) # Indicates how fast to apply box reszizing
    rho = sim.state.N_particles / sim.state.box.volume
    initial_box = sim.state.box
    final_box = hoomd.Box.from_box(initial_box) 
    final_rho = density
    final_box.volume = sim.state.N_particles / final_rho
    box_resize_trigger = hoomd.trigger.Periodic(10)
    box_resize = hoomd.update.BoxResize(box1=initial_box, box2=final_box, variant=ramp, trigger=box_resize_trigger)
    sim.operations.updaters.append(box_resize)
    sim.run(10_000)

    # Equilibrate Simulation
    sim.run(eq_time)
    sim.operations.updaters.remove(box_resize)
    
    
    gsd_writer = hoomd.write.GSD(filename='traj' + str(it) +'.gsd',
                              trigger=hoomd.trigger.Periodic(1000),
                              mode='wb')
    sim.operations.writers.append(gsd_writer)
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermodynamic_properties)
    logger = hoomd.logging.Logger()
    logger.add(thermodynamic_properties)
    gsd_writer.log = logger
    sim.run(production_time)
    
    gsd_writer.flush()