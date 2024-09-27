module InevitableFailure

using JSON
using Plots
using Random
using Statistics
using SparseArrays
using LinearAlgebra

using Base.Threads

export Fault, SimulationState, SubEvent, Earthquake
export simulate_earthquakes

# constants
const EARTH_RAD_KM = 6371.0
const SHEAR_MODULUS = 30e9
const MAGNITUDE_CONSTANT = 9.05


# types
mutable struct Fault
    id::Int
    S::Float64
    S_max::Float64
    S_dot::Float64
    area::Float64  # in square meters
    fid::String
    subsec_id::Int
end

struct SimulationState
    time::Float16
    fault_states::Vector{Float16}
end

struct SubEvent
    fault_id::Int
    δS::Float64
    moment::Float64
end

struct Earthquake
    time::Float64
    sub_events::Vector{SubEvent}
    total_moment::Float64
    magnitude::Float64
end

# basic functions
function calculate_magnitude(moment::Float64)
    return (log10(moment) - MAGNITUDE_CONSTANT) / 1.5
end


function failure_probability(fault::Fault; f_exp=4.0)
    return failure_probability(fault.S, fault.S_max; f_exp=f_exp)
end

function failure_probability(S, S_max; f_exp=4.0)
    return (S / S_max)^f_exp
end

function check_failure(fault::Fault; f_exp=4.0, _rand=-1.0,
)
    if _rand == -1.0
        _rand = rand()
    end
    return _rand < failure_probability(fault; f_exp=f_exp)
end

function _drop_fraction(x, n)
    drop_fraction = 1.0 - (1.0 - x)^n
end


function S_drop(S, S_drop_exp=1.0; _rand=-1.0)
    if _rand == -1.0
        _rand = rand()
    end
    #drop_fraction = _rand ^ S_drop_exp
    drop_fraction = _drop_fraction(_rand, S_drop_exp)
    return S * drop_fraction
end

function reduce_S!(fault::Fault; S_drop_exp=1.0, S_rand=-1.0)
    δS = S_drop(fault.S, S_drop_exp; _rand=S_rand)
    fault.S -= δS
    moment = fault.area * δS * SHEAR_MODULUS
    return δS, moment
end


function update_S!(fault::Fault, δt::Float64)
    fault.S = min(fault.S + fault.S_dot * δt, fault.S_max)
end


function apply_effect!(fault::Fault, effect::Float64)
    fault.S += effect
    if fault.S < 0.0
        fault.S = 0.0
    elseif fault.S > fault.S_max
        fault.S = fault.S_max
    end
end


function distance_scaling(distance, dist_exp=2.0, dist_constant=1.0)
    if distance == 0.0
        return 0.0
    end
    scaled_val = dist_constant / (distance^dist_exp)
    if isnan(scaled_val)
        scaled_val = 0.0
    end
    scaled_val
end


function propagate_effect!(faults, source_index::Int, δS::Float64, adjacency_matrix)
    for j in eachindex(faults)
        effect = adjacency_matrix[j, source_index] * δS
        apply_effect!(faults[j], effect)
    end

end


function handle_earthquake!(host_fault_id, faults, trigger_time::Float64, adjacency_matrix, states;
    cascade=true, f_exp=4.0, S_drop_exp=1.0, S_rand=-1.0, track_state=false)

    fault_0 = faults[host_fault_id]
    sub_events = SubEvent[]

    total_moment = 0.0
    any_failure = true
    current_time = trigger_time

    S_before = fault_0.S
    if track_state
        push!(states, SimulationState(current_time, [fault.S for fault in faults]))
        #@assert states[end].time == current_time
        #@assert states[end].fault_states[host_fault_id] == S_before
    end

    δS, moment = reduce_S!(fault_0; S_drop_exp=S_drop_exp, S_rand=S_rand)
    push!(sub_events, SubEvent(fault_0.id, δS, moment))
    @assert(fault_0.S == S_before - δS)
    total_moment += moment
    propagate_effect!(faults, fault_0.id, δS, adjacency_matrix)
    if track_state
        push!(states, SimulationState(current_time, [fault.S for fault in faults]))
        #@assert states[end].fault_states[host_fault_id] == fault_0.S
    end

    while any_failure
        any_failure = false
        for fault in faults
            if fault.id == fault_0.id
                break
            end
            if check_failure(fault; f_exp=f_exp)
                if cascade
                    any_failure = true
                else
                    any_failure = false
                end
                δS, moment = reduce_S!(fault; S_drop_exp=S_drop_exp, S_rand=S_rand)
                push!(sub_events, SubEvent(fault.id, δS, moment))
                total_moment += moment
                propagate_effect!(faults, fault.id, δS, adjacency_matrix)
                if track_state
                    push!(states, SimulationState(current_time, [fault.S for fault in faults]))
                end
            end
        end
        current_time += 1e-6  # Small time increment for cascading events
    end

    if !isempty(sub_events)
        magnitude = calculate_magnitude(total_moment)
        return Earthquake(trigger_time, sub_events, total_moment, magnitude)
    else
        return nothing  # Return nothing if no sub-events occurred
    end
end


function simulate_earthquakes(faults, distance_matrix, max_time, δt;
    verbose=false, cascade=true, f_exp=4.0,
    S_drop_exp=1.0, S_rand=-1.0, dist_exp=2.0,
    dist_constant=1.0, track_state=false)
    time = 0.0
    earthquakes = Earthquake[]
    states = SimulationState[]

    adjacency_matrix = distance_scaling.(distance_matrix, dist_exp, dist_constant)

    while time < max_time
        if verbose
            @info time
            flush(stdout)
        end
        if track_state
            push!(states, SimulationState(time, [fault.S for fault in faults]))
        end

        for (fault_idx, fault) in enumerate(faults)
            update_S!(fault, δt)
            if check_failure(fault; f_exp=f_exp)
                earthquake = handle_earthquake!(fault_idx, faults, time, adjacency_matrix,
                    states;
                    cascade=cascade, f_exp=f_exp,
                    S_drop_exp=S_drop_exp,
                    S_rand=S_rand,
                    track_state=track_state)
                if earthquake !== nothing
                    push!(earthquakes, earthquake)
                end
            end
        end
        if verbose
            n_eqs = length(earthquakes)
            @info "$n_eqs"
            flush(stdout)
        end
        time += δt
    end

    # Add final state
    push!(states, SimulationState(time, [fault.S for fault in faults]))

    return earthquakes, states
end

# statistics and analytics
function b_mle(magnitudes::Vector{Float64}, M_c::Float64)
    # Filter magnitudes above the cutoff
    magnitudes_above_cutoff = filter(m -> m >= M_c, magnitudes)

    # Check if we have enough data
    if length(magnitudes_above_cutoff) < 2
        error("Not enough magnitudes above the cutoff for a reliable estimate")
    end

    # Calculate mean magnitude
    M_mean = mean(magnitudes_above_cutoff)

    # Calculate b-value
    b = log10(exp(1)) / (M_mean - (M_c - 0.05))

    return b
end

function b_mle(earthquakes::Vector{Earthquake}, M_c::Float64)
    #magnitudes = earthquakes.magnitude
    magnitudes = map(e -> e.magnitude, earthquakes)
    return b_mle(magnitudes, M_c)
end


function b_mle(earthquakes::Vector{Earthquake})
    M_c = minimum(map(e -> e.magnitude, earthquakes))
    return b_mle(earthquakes, M_c)
end

function sim_earthquakes_with_stats(faults, distance_matrix, max_time, δt;
    verbose=false, cascade=true, f_exp=4.0,
    S_drop_exp=1.0, S_rand=-1.0, dist_exp=2.0,
    dist_constant=1.0)

    earthquakes, states = simulate_earthquakes(faults, distance_matrix, max_time,
        δt; verbose=verbose, cascade=cascade,
        f_exp=f_exp, S_drop_exp=S_drop_exp,
        S_rand=S_rand, dist_exp=dist_exp,
        dist_constant=dist_constant
    )
    b_val = b_mle(earthquakes, 6.5)
    return b_val
end

function calc_moment_rate(earthquakes)
    times = map(e -> e.time, earthquakes)
    moments = map(e -> e.total_moment, earthquakes)

    net_time = times[end] - times[1]
    moment_rate = moments[end] / net_time
end

function fault_moment_rate(fault::Fault)
    return fault.area * fault.S_dot * 30e9
end


function get_fault_state(states, index)
    times = map(s -> s.time, states)
    f_state = map(s -> s.fault_states[index], states)
    return (times, f_state)
end

function find_earthquakes_on_fault(earthquakes, fault_id)
    eqs = Earthquake[]
    for eq in earthquakes
        for sub_event in eq.sub_events
            if sub_event.fault_id == fault_id
                push!(eqs, eq)
                break
            end
        end
    end
    return eqs
end

# geometry functions
function point_sphere_to_cart(pos; R=EARTH_RAD_KM)
    lon = pos[1]
    lat = pos[2]
    depth = pos[3]

    r = R + depth

    x = r * cosd(lat) * cosd(lon)
    y = r * cosd(lat) * sind(lon)
    z = r * sind(lat)

    [x, y, z]
end

function point_cart_to_sphere(pos; R=EARTH_RAD_KM)
    x = pos[1]
    y = pos[2]
    z = pos[3]

    point_cart_to_sphere(x, y, z; R=R)
end

function point_cart_to_sphere(x, y, z; R=EARTH_RAD_KM)
    r = sqrt(x^2 + y^2 + z^2)
    lat = asind(z / r)
    lon = atan2d(y, x)

    depth = r - R

    [lon, lat, depth]
end

function min_distance(points1::Vector{Vector{T}}, points2::Vector{Vector{T}}) where {T<:Real}
    min_dist = Inf
    for p1 in points1
        for p2 in points2
            dist = norm(p1 - p2)
            if dist < min_dist
                min_dist = dist
            end
        end
    end
    return min_dist
end

function all_pair_distances!(transformed_points, adj_matrix)
    keyz = collect(keys(transformed_points))
    n = length(keyz)

    @threads for i in 1:n
        for j in (i+1):n
            key1, key2 = keyz[i], keyz[j]
            dist = min_distance(transformed_points[key1], transformed_points[key2])
            if dist == 0.0
                dist = 1.0
            end

            adj_matrix[key1, key2] += dist
            adj_matrix[key2, key1] += dist
        end
    end
end


function all_pair_distances(transformed_points)
    keyz = collect(keys(transformed_points))
    n = length(keyz)
    adj_matrix = zeros(n, n)
    all_pair_distances!(transformed_points, adj_matrix)
    return adj_matrix
end


# oddly-specific IO
function transform_corner_pts(cca_js)
    dd = Dict(
        outer_key => [point_sphere_to_cart(point) for point in outer_value["corner_pts"]]
        for (outer_key, outer_value) in cca_js
    )
    return sort(dd)
end


# Plots
function plot_fault_state(states, index::Int)
    times, f_state = get_fault_state(states, index)
    plot(times, f_state)
end

function plot_fault_state(states, index::Array{Int})
    times = map(s -> s.time, states)
    f_states = [get_fault_state(states, i)[2] for i in index]
    labels = ["Fault $i" for i in index]

    plot(times, f_states, label=labels)
end

function plot_mfd(earthquakes)
    mags = map(e -> e.magnitude, earthquakes)
    sort!(mags)
    plot(mags, reverse!(cumsum(ones(length(mags)))))
    plot!(yscale=:log10, minorgrid=true)
end

function plot_eq_times(earthquakes)
    times = map(e -> e.time, earthquakes)
    n_eqs = collect(1:length(earthquakes))
    plot(times, n_eqs)
end


function plot_cumulative_moment(earthquakes)
    times = map(e -> e.time, earthquakes)
    moments = map(e -> e.total_moment, earthquakes)

    net_time = times[end] - times[1]
    moment_rate = moments[end] / net_time

    println("moment rate: $moment_rate")

    plot(times, cumsum(moments))
end


function calc_system_energy(states, faults)
    energy = zeros(length(states))
    for (i, state) in enumerate(states)
        for (j, fault) in enumerate(faults)
            energy[i] += fault.area * SHEAR_MODULUS * state.fault_states[j]
        end
    end
    return energy
end


function plot_system_energy(states::Vector{SimulationState},
    faults::Vector{Fault},
    earthquakes::Vector{Earthquake};
    plot_eqs=true)
    times = map(s -> s.time, states)
    energy = calc_system_energy(states, faults)
    plt = plot(times, energy, legend=false)

    if plot_eqs
        for eq in earthquakes
            vline!([eq.time], color=:red)
        end
    end
    return plt
end

end # module
