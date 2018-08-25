struct event_log
    n_event::Int64
    n_duration::Int64
    n_timestamp::Array{Int64}
    n_task::Int64
    n_thread::Int64
    name::Array{String}
    tid::Array{String}
    evtype::Array{Int}
    evstart::Array{Float64}
    evend::Array{Float64}
    hover::Array{String}
    task_map::Dict{String,Integer}
    tid_map::Dict{String,Integer}
    timestamp_map::Dict{String,Integer}
end

function read_profile_log(file_name)

    lines = open(file_name) do file
        lines = readlines(file)
    end

    # First few lines have information for the thread ids
    first_line = split(lines[1])
    n_thread = parse(Int, first_line[2])

    tid_map = Dict{String,Integer}()

    for j=1:n_thread
        i = j+1
        splt = split(lines[i])
        tid_map[splt[2]] = parse(Int, splt[3])
    end

    n_header = 1+n_thread

    n_event = length(lines) - n_header

    tid = Array{String}(n_event)
    name = Array{String}(n_event)
    hover = Array{String}(n_event)
    evtype = Array{Int}(n_event)
    evstart = Array{Float64}(n_event)
    evend = Array{Float64}(n_event)

    task_map = Dict{String,Integer}()

    n_task = 0

    n_duration = 0
    n_timestamp = [0,0,0] # spawn_to, spawn_other, steal
    timestamp_map = Dict("spawn_to" => 1, "spawn_other" => 2, "steal" => 3)

    for i=1:n_event
        splt = split(lines[i+n_header])
        tid[i] = splt[2]
        evstart[i] = parse(Float64, splt[4])

        if (splt[3] == "start")
            evtype[i] = 0
            n_duration += 1
            evend[i] = parse(Float64, splt[6])
            name[i] = splt[8]

        elseif (splt[3] == "timestamp")
            evtype[i] = 1
            evend[i] = evstart[i]
            name[i] = splt[6]
            hover[i] = ""
            if (name[i] == "spawn_other" || name[i] == "spawn_to" || name[i] == "steal")
                hover[i] = splt[7] * " "
            end

            loc = timestamp_map[ name[i] ]
            n_timestamp[loc] += 1
        end

        if (! haskey(tid_map, tid[i]))
            n_thread += 1
            tid_map[ tid[i] ] = n_thread
        end

        if (! haskey(task_map, name[i]))
            n_task += 1
            task_map[ name[i] ] = n_task
        end
    end

    tmin = minimum(evstart)
    evstart -= tmin
    evend -= tmin

    evstart /= 1e6
    evend /= 1e6

    return event_log(n_event,n_duration,n_timestamp,n_task,n_thread,
        name,tid,
        evtype,evstart,evend,hover,
        task_map,tid_map,timestamp_map),
        minimum(evstart), maximum(evend), n_event
end

function build_profile_plot(ev, tmin, tmax)

    rectangles = Array{ PlotlyBase.PlotlyAttribute{Dict{Symbol,Any}} }(ev.n_duration)

    n_timestamp_max = maximum(ev.n_timestamp)
    x_ts = Array{Float32}(n_timestamp_max,3)
    y_ts = Array{Float32}(n_timestamp_max,3)
    hovertext_ts = Array{String}(n_timestamp_max,3)

    x_ht = Array{Float32}(ev.n_duration)
    y_ht = Array{Float32}(ev.n_duration)
    hovertext_ht = Array{String}(ev.n_duration)

    i_duration = 0
    i_timestamp = [0,0,0]
    i_end = findlast(x -> x <= tmax, ev.evend)
    i_end = max(1, i_end) # At least one entry should be kept

    for i=1:i_end
        thread_id = ev.tid_map[ ev.tid[i] ]
        y0 = thread_id - 0.45
        x0 = ev.evstart[i]
        task_name = ev.task_map[ ev.name[i] ]
        x1 = ev.evend[i]

        if (ev.evtype[i] == 0)
            rect_color = @sprintf("hsla(%3d, 100%%, 50%%, 0.4)",360.0*task_name/(1+ev.n_task))
            if (ev.name[i] == "overhead")
                rect_color = "rgba(128, 128, 128, 0.4)"
            elseif (ev.name[i] == "dependency")
                rect_color = "rgba(192, 192, 192, 0.4)"
            elseif (ev.name[i] == "wait")
                rect_color = "rgba(256, 256, 256, 0.4)"
            end
            rect = attr(xref="x",yref="y",x0=x0,y0=y0,x1=x1,y1=y0+0.9,
            line_width=0.5,line_color="grey",fillcolor=rect_color)
            rect[:shapes_type] = "rect"

            i_duration += 1
            rectangles[i_duration] = rect

            x_ht[i_duration] = 0.5*(x0+x1)
            y_ht[i_duration] = thread_id
            hovertext_ht[i_duration] = ev.name[i] * " e " * @sprintf("%.3fms",x1-x0) * @sprintf(" t %.2fms",x0)

        elseif (ev.evtype[i] == 1)
            x1 = x0
            loc = ev.timestamp_map[ev.name[i]]
            i_timestamp[loc] += 1
            x_ts[i_timestamp[loc],loc] = x0
            if thread_id == -1
                y_ts0 = thread_id
            else
                y_ts0 = thread_id - 0.45
            end
            y_ts[i_timestamp[loc],loc] = y_ts0
            hovertext_ts[i_timestamp[loc],loc] = ev.name[i] * " " * ev.hover[i] * @sprintf("t %.2fms",x0)
        end
    end

    trace = [
        scatter(x=x_ht[1:i_duration],y=y_ht[1:i_duration],
            hovertext=hovertext_ht[1:i_duration],
            mode="markers",marker_opacity=0,name="task",showlegend=false),
        scatter(x=x_ts[1:i_timestamp[1],1], y=y_ts[1:i_timestamp[1],1],
            hovertext=hovertext_ts[1:i_timestamp[1],1],
            mode="markers",marker_symbol="circle-open",marker_size=12,name="spawn_to"),
        scatter(x=x_ts[1:i_timestamp[2],2], y=y_ts[1:i_timestamp[2],2],
            hovertext=hovertext_ts[1:i_timestamp[2],2],
            mode="markers",marker_symbol="square-open",marker_size=12,name="spawn_other"),
        scatter(x=x_ts[1:i_timestamp[3],3], y=y_ts[1:i_timestamp[3],3],
            hovertext=hovertext_ts[1:i_timestamp[3],3],
            mode="markers",marker_symbol="diamond-open",marker_size=12,name="steal")
    ]

    dt = tmax - tmin
    margin_t = dt*0.01
    layout = Layout(yaxis_zeroline=false,yaxis_ticksuffix="  ",
        hovermode="closest",
        shapes=rectangles[1:i_duration],
        xaxis_title="time [ms]", yaxis_title="thread id",
        xaxis_range=[tmin - margin_t, tmax + margin_t],
        yaxis_dtick=1, yaxis_range=[-1.9, ev.n_thread - 1.1]
    )

    return plot(trace,layout)
end

function build_profile_plot(ev)
    build_profile_plot(ev, minimum(ev.evstart), maximum(ev.evend))
end

function build_profile_plot(ev, n_event_plot)
    tmax = ev.evend[ min(length(ev.evend), n_event_plot) ]
    build_profile_plot(ev, minimum(ev.evstart), tmax)
end

function profiler_plot(file_name)
    ev, tmin, tmax, n_event = read_profile_log(file_name)
    return build_profile_plot(ev, tmin, tmax)
end
