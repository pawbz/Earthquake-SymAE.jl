### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 7f2a30cc-eaab-49e7-ba6b-145d4e00f95c
using DrWatson

# ╔═╡ a59c278e-8779-48d0-9c82-4cf2aad0e471
using CUDA,
    cuDNN,
Enzyme,
    Flux,
    Distances,
    CSV,
    JLD2,
    Random,
    MLUtils,
    DSP,
    ProgressLogging,
    Statistics,
    LinearAlgebra,
    PlutoLinks,
    PlutoUI,
    PlutoHooks,
    PlutoPlotly,
    FFTW,
    Healpix,
    PlotlyKaleido,
    ColorSchemes,
    TSne,
    HDF5,
	StatsBase,
    Optimisers

# ╔═╡ 5732782d-c8f5-42d2-a94b-dfcc5cba0a80
using Metalhead

# ╔═╡ 41370b84-a791-4e5f-810c-d5ae458679a1
using Dates, BlackBoxOptim, Metaheuristics

# ╔═╡ aa9d5025-ad7f-4f2a-9c41-8cf1fe72db90
using ParameterSchedulers

# ╔═╡ 9e8a8120-a2d0-4898-b5f4-bd69b722fd39
TableOfContents(include_definitions=true)

# ╔═╡ 974e1d57-1cc7-4258-8ff1-f79ca55e7c9f
xpu = gpu

# ╔═╡ 36d58800-cd47-4a74-aec6-b61306ec0136
md"## Add synthetic"

# ╔═╡ 19c9e110-e09a-4ec1-8aaf-0507236912e5
files_syn = filter(x->occursin("jld2", x), readdir("../multiple_scattering_results_save1/save3", join=true))

# ╔═╡ c48e61dd-12e7-4eb8-aaf0-9036d7890b5e
avgSspectrum = cat(exp.(-0.01*(collect(1:101).-45.0).^2), dims=2);

# ╔═╡ 9cf663af-d59a-4c2b-956e-038c457fd5d2
begin
    Deq_train_syn = []
	Deq_s_syn = []
	Deq_train_g_syn = []
    Deq_test_syn = []
	Deq_test_g_syn = []
    Deqnames_syn = []
    for f in files_syn
            
				data = circshift(DSP.resample(Float32.(load(f)["d"]), 0.5, dims=1), (45, 0))[1:200, :]
				noise = irfft(avgSspectrum .* rfft(randn(size(data)...)), size(data, 1))
				data = data .+ 0.00001 * noise

                data = normalise(data, dims = 1)
                D1, D2 = splitobs((data, load(f)["g"]), at = (0.9), shuffle = true)
                    push!(Deq_train_syn, getobs(D1[1]))
					push!(Deq_train_g_syn, getobs(D1[2]))
                    push!(Deq_s_syn, getobs(Float32.(load(f)["s"])))
                    push!(Deq_test_syn, getobs(D2[1]))
                    push!(Deq_test_g_syn, getobs(D2[2]))
                    push!(Deqnames_syn, basename(f))
    end
	Dtrain_syn = xpu(Deq_train_syn)
	Dtest_syn = xpu(Deq_test_syn)
end

# ╔═╡ 7629650d-95eb-48ff-bc86-fd9e2249d1ea
PlutoPlotly.plot(PlutoPlotly.heatmap(z=randobs(Deq_train_syn)))

# ╔═╡ b46bd074-ad00-4cc4-9967-b1952f4931d3
md"# Training EQ SymAE"

# ╔═╡ 2cd412fd-c57a-4caf-b73f-ac926ea47f8c
function to_length_192(D)
	return map(D) do d
		d[9:200, :]
	end
end

# ╔═╡ c5f7712f-d35a-49ee-a02b-4a96b04cbae3
reload_network_button = @bind reload_network CounterButton("Reload Network")

# ╔═╡ 676642c4-8205-4523-b069-787be1040a5e
md"""
## Train and Save
"""

# ╔═╡ 1e85c1c8-ed8d-44b9-8d3e-025b5014b97b
Threads.nthreads()

# ╔═╡ 58eaa0a2-1289-4629-bad4-2cdedde0dfc2


# ╔═╡ f4df92a7-1501-41c7-bda5-777ffeefe488
# mean(map(data.Dtrain) do D
# 	symae.loss_mse(symae_model.model, D)
# end)

# ╔═╡ 7771d130-0a8a-4803-8a6b-966533c887f1
CUDA.argmax(randn(10,10,2), dims=1)

# ╔═╡ 4f7193f0-b92d-45dc-a449-7fe189eb0ca1
Flux.onehotbatch(cat([1], dims=2), 1:2)

# ╔═╡ 927200b1-d4fc-4004-a468-f599729e178c
reload_network_button

# ╔═╡ 5adec3e2-acd4-4895-bb4e-d5ebf7bde0c2
@bind isyn Slider(0:20, show_value=true)

# ╔═╡ def83d24-1dea-4528-b85b-fc25fb43abc1
@bind envelo_flag CheckBox()

# ╔═╡ 45b2a58f-fbd6-44ee-aa3c-005a45086c54
env = x -> envelo_flag ? abs.(hilbert(x)) : x

# ╔═╡ 2cb046b7-4bec-4bed-bdaa-a3cc260574c7
plot(env(Deq_s_syn[end-isyn]))

# ╔═╡ b5703c40-e188-4a74-abb2-c1a9b1a19079
plot((mean(env(Deq_train_syn[end-isyn]), dims=2)))

# ╔═╡ b88703d2-72ed-41be-a799-732166740855
PlotlyKaleido.start()

# ╔═╡ 2033dffb-7e8c-4f27-ac4a-c0c9522eb5c8
md"## Saving"

# ╔═╡ 0b4f54c4-0a84-466c-9c19-4cfe68e82a98
tgrid=range(-100, 100, length=200)

# ╔═╡ 8339c5a0-6fad-451b-b4df-ed22588d806a
md"## Appendix"

# ╔═╡ af479841-ac07-4b8d-b58d-7575a94f9be4
import MLUtils: sample

# ╔═╡ 818c44ef-86bf-410d-b698-8cff6c6969b1
CUDA.device!(1)

# ╔═╡ d4034569-cd45-4a98-94d2-0de39491201e
symae = @ingredients("../SymAE_architecture.jl")

# ╔═╡ 5abc5ef3-93a0-4eef-b6bf-7e1ddf16c232
# symae_parameters = (; nt = 200,
#     p = 30,
#     q = 30,
#     ntau = ntau,
#     nsteps = 100,
#     batchsize = 32,
#     beta = 1.0,
#     nhidden_dense = 1024,
#     nlayer_dense = 3,
#     network_type = :conv,
# 	spatial_transformer_gamma = 1f2,
#     transformer = :spatial,
# 	seed=nothing
# )
symae_parameters = (; nt = 192,
    p = 30,
    q = 30,
	k = 1,
    network_type = symae.ConvAE(),
    transformer = :spatial,
	seed=nothing
)

# ╔═╡ f5eb2333-be02-4891-9bde-c5bc917278d4
symae_para = symae.SymAE_Para(; symae_parameters...)

# ╔═╡ 988f1f0e-e6f9-43a2-998f-e8373185b26f
symae_model, loss_history = @use_memo([reload_network]) do
    reload_network
    return symae.get_symae(symae_para)
end

# ╔═╡ 76d98574-1ca1-4652-ac52-fa3e4d5f4608
symae_model

# ╔═╡ 7e525230-9c74-442c-987d-a1cabec96b7a
symae_model(xpu(randn(192,3)), Val(:coherent))

# ╔═╡ 68bcf7d0-4163-45f4-ab67-3ef6ca8c57db
begin
	# trained/
	
	symae.plot_loss_history(loss_history)
end

# ╔═╡ 25360328-f93a-4ab4-93c9-73c5808a8847
eqdataload = @ingredients("EqDataLoad.jl")

# ╔═╡ 62655df3-0e96-4567-b882-9b6babc972f6
# data_all = xpu(eqdataload.data_pixel_all); # both P and pP{}
data_all = xpu(eqdataload.data_pixel); # both P and pP{}

# ╔═╡ eb429d30-e5ad-4d27-9cbb-7c0ad2315948
PlutoPlotly.plot(PlutoPlotly.heatmap(z=randobs(data_all.Dtrain)))

# ╔═╡ 1be70e89-bf9d-4702-be40-8ed119d257ec
data_all

# ╔═╡ 1ccc0ec1-798b-42cb-a91c-5f6b06820ca7
begin

	# kidx_okt1 = (findall(x -> occursin("20021117", x), data_all.Dnames))
	# kidx_okt1 = (findall(x -> occursin(selected_eq_name_train, x), data_all.Dnames))
	kidx_okt1 = 1:length(data_all.Dtrain)
	data = (; Dtrain = vcat(to_length_192(data_all.Dtrain[kidx_okt1]), to_length_192(Dtrain_syn)), Dtest = to_length_192(data_all.Dtest[kidx_okt1]), 
	Deq_train_stn = data_all.Deq_train_stn[kidx_okt1], 
	Deq_test_stn = data_all.Deq_test_stn[kidx_okt1],
	Dnames = vcat(data_all.Dnames[kidx_okt1], map(x->string("syn", x), 1:length(Dtrain_syn))))
	# data.Deq_train_stn = data.Deq_train_stn[kidx_okt1]
	# Dnames = Dnames[kidx_okt1]
end

# ╔═╡ 235ec0aa-5ef6-47ce-85b2-b96407266cd3
begin
    ntau = min(minimum(getindex.(size.(data.Dtrain), 2)), 50)
end

# ╔═╡ 2fd9a503-cf69-4c80-8af1-ac68a7a18886
trained = @use_memo([]) do

	# update = symae.update_alternating_transformer_batchviews
	update = symae.update
   training_para=symae.Training_Para(nepoch=5, gamma=1f2, initial_learning_rate=0.001, temperature=1f0)
    update(symae_model, loss_history, data.Dtrain, data.Dtest, training_para)
# 	 training_para=symae.Training_Para(nepoch=50, gamma=1f2, initial_learning_rate=0.0001, temperature=0.5f0)
# 	update(symae_model, loss_history, data.Dtrain, data.Dtest, training_para)
# training_para=symae.Training_Para(nepoch=100, gamma=1f2, initial_learning_rate=0.00001, temperature=0.25f0)
# 	update(symae_model, loss_history, data.Dtrain, data.Dtest, training_para)
	   	   
    # symae.update_alternating_transformer_batchviews(symae_model.model, loss_history, data.Dtrain, data.Dtest, symae_para, nepoch = 100, nprint=1, learning_rate=0.0001)
    # symae.update_alternating_transformer_batchviews(symae_model.model, loss_history, data.Dtrain, data.Dtest, symae_para, nepoch = 100, nprint=1, learning_rate=0.0001)
	# symae.update_alternating_transformer(symae_model.model, loss_history, data.Dtrain, data.Dtest, symae_para, nepoch = 200, nprint=1, learning_rate=0.0001)
	# symae.update_alternating_transformer(symae_model.model, loss_history, data.Dtrain, data.Dtest, symae_para, nepoch = 100, nprint=1, learning_rate=0.00001)
	
# # kjkkkk
	
# 	timestamp = now()
# 	savename_para = savename(symae_parameters)
# 	jldsave(joinpath("SavedModels", "syn_model-$(timestamp)-$(savename_para).jld2"), model_state = Flux.state(cpu(symae_model.model)))
# jldsave(joinpath("SavedModels", "syn_para-$(timestamp)-$(savename_para).jld2"); symae_parameters)
# 	jldsave(joinpath("SavedModels", "syn_loss_history-$(timestamp)-$(savename_para).jld2"); loss_history)
	
    randn(), loss_history
end

# ╔═╡ 818f312f-2370-4606-bc92-579030e9cb97
let
    d, dname = randobs((data.Dtrain, data.Dnames), 1)
   S = symae_model.transformerb(d[1])
    plot(
        histogram(x = cpu(vec(S.shifts)), xbins_size = 0.005),
        Layout(title = dname[1], xaxis = attr(range = (-1, 1))),
    )
end

# ╔═╡ b2d8ae0d-c4c8-46fa-beba-eeceed8e1321
argmax(map(x->size(x, 2), data.Dtrain))

# ╔═╡ 6382acfb-f9de-4ff3-975c-ca2abf724493
stf_syn, ideal_seismograms_syn, nuisance_optim_loss_syn, shifted_ideal_seismograms_syn = @use_memo([]) do 	
	symae.get_coherent_information(symae_para, symae_model, data.Dtrain[end-20:end], data.Dtrain[end-20:end],  initialize_within_state=true, N=10, nepochs=100, alpha=1, learning_rate=0.01)
end

# ╔═╡ 0fc67113-d195-4ceb-894c-f633d4c24b0c
plot((env(stf_syn[end-isyn])))

# ╔═╡ 4483c965-6782-4195-ba92-29beb9a7c635
plot(cpu(shifted_ideal_seismograms_syn[:, :, end-isyn]))

# ╔═╡ b953dca6-8d64-473b-9d86-705a5ee443ee
plot(cpu(ideal_seismograms_syn[:, :, end-isyn]))

# ╔═╡ d6088ec8-43a5-42ee-b3e6-e890c8a3de4f
data.Dnames

# ╔═╡ c9b8fae4-7199-4e4d-a8ee-6bca22770bb2
data.Dnames

# ╔═╡ f82e40a8-1321-4ffe-982e-e72aacdae538
eqnames = unique(map(data_all.Dnames) do eq
	join(split(eq, "_")[1:end-1], "_")
end)

# ╔═╡ 2a36b358-9fd6-4d06-8078-c467a6b070a3
@bind selected_eq_name_train confirm(Select(eqnames, default=randobs(eqnames)))

# ╔═╡ f10fbe7f-e161-4d73-9e27-56d32860b42d
@bind selected_eq_name confirm(Select(eqnames, default=selected_eq_name_train))

# ╔═╡ 86c84c4b-6449-4a7c-9ae8-6408ded55b70
kidx1 = findall(x -> occursin(selected_eq_name, x), data.Dnames)

# ╔═╡ 948e21ce-b897-4f69-adcd-ed176761798e
data.Dnames[kidx1]

# ╔═╡ 7513d195-e637-4e68-bdfa-7d879c5ca326
argmax(map(x->size(x, 2), data.Dtrain[kidx1]))

# ╔═╡ cb7c950d-40dc-4890-9e8f-20b8b77448cd
begin
	kidx_max_rec = kidx1[argmax(map(x->size(x, 2), data.Dtrain[kidx1]))]
	# kidx_max_rec = 32
end

# ╔═╡ 143fe031-187f-41b8-95ae-f1eee558ca31
kidx_max_rec

# ╔═╡ 2b8e3682-10e6-426a-8c1d-ae774c270844
stf, ideal_seismograms, nuisance_optim_loss, shifted_ideal_seismograms = @use_memo([]) do 	
	symae.get_coherent_information(symae_para, symae_model, data.Dtrain[kidx_max_rec], data.Dtrain[kidx1], data.Dtrain[kidx1], N=10, nepochs=100, alpha=0.1, learning_rate=0.1, kopt=1)
end

# ╔═╡ 9d2450b7-e96e-48f9-9100-9b862d0074d5
plot((cpu(shifted_ideal_seismograms)))

# ╔═╡ b695c9d5-cf22-44eb-a877-0701a88f5f1d
plot(mean(cpu(shifted_ideal_seismograms), dims=2))

# ╔═╡ 3ca58bfb-8b77-4354-b2e3-e6decf3f92e9
plot(((stf[3])))

# ╔═╡ 584ae1ac-7d69-4e17-b2bd-30736d52bb90
plot((stf[1]))

# ╔═╡ a8cea190-fdd2-496e-83d5-b186a1adbecd
plot(cpu(ideal_seismograms))

# ╔═╡ 229949e4-aae3-4c00-931f-382795f38b13
plot(nuisance_optim_loss)

# ╔═╡ e2df0cc3-f202-4366-aa7c-bd18046151e6
stf2, ideal_seismograms2, nuisance_optim_loss2, shifted_ideal_seismograms2 = @use_memo([]) do 	
	symae.get_coherent_information(symae_para, symae_model, data.Dtrain[kidx_max_rec], data.Dtrain[kidx_max_rec:kidx_max_rec], data.Dtrain[kidx_max_rec:kidx_max_rec+10], N=10, nepochs=100, alpha=0.1, learning_rate=0.1, kopt=1)
end

# ╔═╡ fdb76d1a-e114-4126-a214-cf9d00a80c09
plot(((stf2[1])))

# ╔═╡ 11ed37bf-aa3c-4bb4-82ce-0a0961247e4c
stf3, ideal_seismograms3, nuisance_optim_loss3, shifted_ideal_seismograms3 = @use_memo([]) do 	
	symae.get_coherent_information(symae_para, symae_model, data.Dtrain[kidx_max_rec:kidx_max_rec+10], data.Dtrain[kidx_max_rec:kidx_max_rec+10], N=10, nepochs=100, alpha=0.1, initialize_within_state=false, learning_rate=0.1, kopt=1)
end

# ╔═╡ d622fc67-7f2e-4459-8e30-aac42a312ad3
plot(((stf3[1])))

# ╔═╡ c3dce92a-46dc-4af8-816b-c1880858b85b
envelope(x) = abs.(hilbert(x));
# envelope(x) = x

# ╔═╡ 63b4fb92-783f-445e-919e-b3a3519dd06a
md"""
### Plots
"""

# ╔═╡ 98863e40-d3d1-4eb6-9325-67373f3f973e
function plot_loss(d; title = "", labels = fill(nothing, length(d)))
    trace = [
        PlutoPlotly.scatter(y = cpu(dd), mode = "lines", name = label) for
        (dd, label) in zip(cpu(d), labels)
    ]
    layout = Layout(
        title = title,
        legend = attr(x = 0, y = -0.3, traceorder = "normal", orientation = "h"),
        yaxis = attr(gridwidth = 1),
        yaxis_type = "log",
        height = 500,
    )
    PlutoPlotly.plot(trace, layout)
end

# ╔═╡ 8e5e4a99-35c5-426a-8440-bf773be6f252
function plot_stf(stf, names, scale=2)
	# Extract bin numbers from names
    # Extract the numeric bin indices
    bin_indices = [parse(Int, last(split(bin, "_"))) for bin in names]
    
    # Get the sorted permutation of bin indices
    sorted_indices = sortperm(bin_indices)
    
    # Reorder `names` based on the sorted indices
    sorted_names = names[sorted_indices]
	sorted_stf = stf[sorted_indices]
	
	sorted_bin_indices = bin_indices[sorted_indices]
	max_bin_number=maximum(sorted_bin_indices)
	min_bin_number=minimum(sorted_bin_indices)
	n_bins = max_bin_number - min_bin_number + 1  # Number of unique bins
	color_scheme = ColorSchemes.darkrainbow
	color_samples = [get(color_scheme, i / (n_bins - 1)) for i in 0:n_bins-1]
	
	# Map bins to their corresponding colors
	bin_to_color = Dict(bin => color_samples[bin - min_bin_number + 1] for bin in min_bin_number:max_bin_number)


	
	eqname = first(split(first(sorted_names), "_"))
    av = 0
    scatter_plots = map(enumerate(sorted_names)) do (ibin, bin)
        idx_bin = parse(Int, last(split(bin, "_")))
        sout = (Array(sorted_stf[ibin])) #* 2
        av = av .+ sout
        previous_yoffset = 0
        current_yoffset = (scale * (length(sorted_names[1:ibin]) - 1))
		
		# Get the color for the bin
        #bin_color = get_bin_color(idx_bin, color_gradient)
		bin_color = bin_to_color[idx_bin]

        return PlutoPlotly.scatter(
            x = tgrid,
            y = sout .+ current_yoffset .+ previous_yoffset,
			fillcolor = bin_color,
			line = attr(color = "black"),
            fill = "tonexty",
            mode = "lines",
            textposition = "top",
        )
    end

    annotations = map(enumerate(sorted_names)) do (ibin, bin)
        idx_bin = parse(Int, last(split(bin, "_")))
        angles = broadcast(pix2angRing(Resolution(4), idx_bin)) do x
            floor(Int, rad2deg(x))

            # print(idx_bin)
        end
		

        current_yoffset = (scale * (length(sorted_names[1:ibin]) - 1))
        previous_yoffset = 0

        return attr(
            x = -40,
            y = current_yoffset .+ previous_yoffset,
            xref = "x",
            yref = "y",
            text = string(idx_bin, ": ", angles),
            showarrow = false,
            font = attr(size = 18, color = "black"),
            align = "center",
            bordercolor = "#c7c7c7",
            borderwidth = 2,
            borderpad = 4,
            bgcolor = "white",
            opacity = 1,
        )
    end

    # phase_times = [ pP, sP ]
    # phase_names = [ "pP", "sP"]

    # map(phase_times, phase_names) do pt, pn
    #     push!(annotations, attr(
    #         x=pt,
    #         y=-1.2,
    #         xref="x",
    #         yref="y",
    #         text=pn,
    #         textangle=90,
    #         showarrow=false,
    #         font=attr(
    #             size=18,
    #             color="red"
    #         ),
    #         align="center",
    #         # bordercolor="white",
    #         # borderwidth=2,
    #         # borderpad=4,
    #         # bgcolor="white",
    #         # opacity=1
    #     ))
    # end
    average = av / length(sorted_names)
    current_yoffset1 = (scale * (length(sorted_names) + 0.1))
    average_plot = PlutoPlotly.scatter(
        x = tgrid,
        y = average .+ current_yoffset1,
        mode = "lines",
        textposition = "top",
        line = attr(color = "black"),
    )
    push!(
        annotations,
        attr(
            x = -45,
            y = current_yoffset1,
            xref = "x",
            yref = "y",
            text = "Mean",
            showarrow = false,
            font = attr(size = 18, color = "black"),
            align = "center",
            bordercolor = "#c7c7c7",
            borderwidth = 2,
            borderpad = 4,
            bgcolor = "white",
            opacity = 1,
        ),
    )
    current_yoffset2 = (scale * (length(sorted_names) + 2))
    # path_plot1 = PlutoPlotly.scatter(
    #     x = tgrid,
    #     y = vec(Array(selected_path)) .+ current_yoffset2,
    #     mode = "lines",
    #     line = attr(color = "blue"),
    #     textposition = "top",
    # )
    # path_plot2 = scatter(
    #     x = tgrid,
    #     y = vec(Array(selected_path_hat)) .+ current_yoffset2,
    #     mode = "lines",
    #     line = attr(color = "red", dash = "dash"),
    #     textposition = "top",
    # )

	#uncomment this to get the "1" you want in the plot in text side
    #push!(
    #    annotations,
    #    attr(
    #       x = -80,
    #        y = current_yoffset2,
    #        xref = "x",
    #        yref = "y",
    #        # text=split(kstn," ")[2],
    #       text = istn,
    #        showarrow = false,
    #        font = attr(size = 18, color = "black"),
    #        align = "center",
    #        bordercolor = "#c7c7c7",
    #        borderwidth = 2,
    #        borderpad = 4,
    #        bgcolor = "white",
    #        opacity = 1,
    #    ),
    #)

    # vertical_pP = phase_times[1]
    # vertical_sP = phase_times[2]



    # Create a vertical line trace pP
    # vertical_linepP = scatter(x=[vertical_pP, vertical_pP], y=[0, (5 * (length(names) +2))],mode="lines", line=attr(color="black", dash="dash"))

    # # Create a vertical line trace sP
    # vertical_linesP = scatter(x=[vertical_sP, vertical_sP], y=[0, (5 * (length(names) +2))],mode="lines", line=attr(color="black", dash="dash"))


    p = PlutoPlotly.plot(
        vcat(scatter_plots, average_plot),
        Layout(
            showlegend = false,
            annotations = annotations,
            font_family = "Computer Modern",
            font_color = "black",
            font = attr(family = "Computer Modern", size = 22),
            width = 600,
            height = 300 + 2 * sum(length.(sorted_names)),
            template = "none",
            yaxis_showgrid = false,
            xaxis_range = (-50, 60),
            xaxis = attr(
                title = "Time Relative to PREM P (s)",
				standoff = 0,  # Distance from the axis
                tickfont = attr(size = 22),
                nticks = 10,
                gridwidth = 1,
                gridcolor = "Gray",
                font_family = "Computer Modern",
                font_color = "black",
                # xaxis_range = (-50, 100),
            ),
			title = attr(
                font = attr(size = 30),
				#text = "$(eqname)",
                x = 0.5,  # Center title horizontally
                xanchor = "center",
                y = 0.95,  # Position title closer to the top
                yanchor = "top"
            ),
            # title=attr(text=eqname,font = attr(size = 30), x=-100,),
            yaxis = attr(showticklabels = false),
            legend = attr(font = attr(size = 18)),
			margin = attr(b = 150),
        ),
    )
end

# ╔═╡ 4bea2566-9b11-440b-959d-f131c79dc47e
p1 = plot_stf(normalise.(map(x->vec(mean(envelope(x[:,:]), dims=2)), stf), dims=1), data.Dnames[kidx1], 3)

# ╔═╡ 3e6d4563-1e35-4adb-84cb-2cb5c6eacea4
PlotlyKaleido.savefig(p1, "/home/paper//NASData/EQData/Deep-Earthquakes/RAW Envelopes/USVS_P_new/plot_$(ueqnames[idx]).png", height=950, width=450, scale=8)

# ╔═╡ c04c2211-a87e-41ee-a42a-e2ea03c9d0bc
p2 = plot_stf(normalise.(map(x->vec(mean(envelope(cpu(x[:,:])), dims=2)), data.Dtrain[kidx1]), dims=1), data.Dnames[kidx1], 3)

# ╔═╡ 36f93d83-45b3-45f3-a881-ac20f8cb2c6e
md"### UI"

# ╔═╡ fb981c7e-24bc-4983-abc1-182989122562
function eq_input()

    return PlutoUI.combine() do Child

        inputs = [
            md""" Eq 1: name $(
            	Child("eq1", TextField(default="okhotsk1"))
            )
            	station: $(
            	Child("eq1stn", TextField())
            )
            """,
            md""" Eq 2: name $(
            	Child("eq2", TextField(default="chile"))
            )
            station: $(
            	Child("eq2stn", TextField())
            )
            """,
        ]

        md"""
        ##### Quake Select
        $(inputs)
        """
    end
end

# ╔═╡ da15d3eb-e776-43f2-8026-a06b61c33965
resample_selected_pathui1 =
    @bind resample_selected_path1 CounterButton("Resample Eq1 Selected Path")

# ╔═╡ 4501fdf1-42d6-4703-bcf1-3be915b2931b
resample_selected_pathui2 =
    @bind resample_selected_path2 CounterButton("Resample Eq2 Selected Path")

# ╔═╡ 8d501f0b-f45c-4e26-9de8-fa1ff0eb2b2f
resample_selected_pathui = [resample_selected_pathui1, resample_selected_pathui2];

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BlackBoxOptim = "a134a8b2-14d6-55f6-9291-3336d3ab0209"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
DrWatson = "634d3b9d-ee7a-5ddf-bec9-22491ea816e1"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
Healpix = "9f4e344d-96bc-545a-84a3-ae6b9e1b672b"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
Metaheuristics = "bcdb8e00-2c21-11e9-3065-2b553b22f898"
Metalhead = "dbeba491-748d-5e0e-a39e-b530a07fa0cc"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
ParameterSchedulers = "d7d3b36b-41b8-4d0d-a2bf-768c6151755e"
PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
TSne = "24678dba-d5e9-5843-a4c6-250288b04835"
cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[compat]
BlackBoxOptim = "~0.6.3"
CSV = "~0.10.15"
CUDA = "~5.8.2"
ColorSchemes = "~3.29.0"
DSP = "~0.8.4"
Distances = "~0.10.12"
DrWatson = "~2.18.0"
Enzyme = "~0.13.23"
FFTW = "~1.9.0"
Flux = "~0.16.4"
HDF5 = "~0.17.2"
Healpix = "~4.2.4"
JLD2 = "~0.5.15"
MLUtils = "~0.4.8"
Metaheuristics = "~3.4.0"
Metalhead = "~0.9.5"
Optimisers = "~0.4.6"
ParameterSchedulers = "~0.4.3"
PlotlyKaleido = "~2.3.0"
PlutoHooks = "~0.0.5"
PlutoLinks = "~0.1.6"
PlutoPlotly = "~0.6.3"
PlutoUI = "~0.7.66"
ProgressLogging = "~0.1.5"
StatsBase = "~0.34.5"
TSne = "~1.3.0"
cuDNN = "~1.4.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "6894e8df2bffc7275c755e12ee082b79e3e30cad"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "4e25216b8fea1908a0ce0f5d87368587899f75be"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "b5bb4dc6248fde467be2a863eb8452993e74d402"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.1"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "3b642331600250f592719140c60cf12372b82d66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.5.1"

[[deps.BSON]]
git-tree-sha1 = "4c3e506685c527ac6a54ccc0c8c76fd6f91b42fb"
uuid = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
version = "0.3.9"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "26f41e1df02c330c4fa1e98d4aa2168fdafc9b1f"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.4"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.BlackBoxOptim]]
deps = ["CPUTime", "Compat", "Distributed", "Distributions", "JSON", "LinearAlgebra", "Printf", "Random", "Requires", "SpatialIndexing", "StatsBase"]
git-tree-sha1 = "9c203a2515b5eeab8f2987614d2b1db83ef03542"
uuid = "a134a8b2-14d6-55f6-9291-3336d3ab0209"
version = "0.6.3"

    [deps.BlackBoxOptim.extensions]
    BlackBoxOptimRealtimePlotServerExt = ["HTTP", "Sockets"]

    [deps.BlackBoxOptim.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
    Sockets = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CFITSIO]]
deps = ["CFITSIO_jll"]
git-tree-sha1 = "8c6b984c3928736d455eb53a6adf881457825269"
uuid = "3b1b4be9-1499-4b22-8d78-7db3344d1961"
version = "1.7.2"

[[deps.CFITSIO_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "LibCURL_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "15e80be798d7711411f4ac4273144cdb2a89eb2f"
uuid = "b3e40c51-02ae-5482-8a39-3ace5868dcf4"
version = "4.6.2+0"

[[deps.CPUTime]]
git-tree-sha1 = "2dcc50ea6a0a1ef6440d6eecd0fe3813e5671f45"
uuid = "a9c8d775-2e2e-55fc-8582-045d282d599e"
version = "1.0.0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics", "demumble_jll"]
git-tree-sha1 = "b8ae59258f3d96ce75a00f9229e719356eb929d6"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.8.2"

    [deps.CUDA.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SparseMatricesCSRExt = "SparseMatricesCSR"
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.CUDA.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "18afa851ed10552e6df25dfaa7ef450104ae73d4"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.13.1+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "33576c7c1b2500f8e7e6baa082e04563203b3a45"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.3.5"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "b5c173a64f9f4224a82fdc26fda8614cb2ecfa27"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.17.1+0"

[[deps.CUDNN_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "12ab2a3e7d1e5f523e95b94cd9334ebcf5be220e"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "9.10.0+0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "224f9dc510986549c8139def08e06f78c562514d"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.72.5"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "06ee8d1aa558d2833aa799f6f0b31b30cada405f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.2"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "062c5e1a5bf6ada13db96a4ae4749a4c2234f521"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.9"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Combinatorics]]
git-tree-sha1 = "8010b6bb3388abe68d95743dcbea77650bb2eddf"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.3"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "3a3dfb30697e96a440e4149c8c51bf32f818c0f3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.17.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "5989debfc3b38f736e69724818210c67ffee4352"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.4"

    [deps.DSP.extensions]
    OffsetArraysExt = "OffsetArrays"

    [deps.DSP.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3e6d038b77f22791b8e3472b7c633acea1ecac06"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.120"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DrWatson]]
deps = ["Dates", "FileIO", "JLD2", "LibGit2", "MacroTools", "Pkg", "Random", "Requires", "Scratch", "UnPack"]
git-tree-sha1 = "17c9db646d6cb3f68a90053bf46faf5312d13b36"
uuid = "634d3b9d-ee7a-5ddf-bec9-22491ea816e1"
version = "2.18.0"

[[deps.Enzyme]]
deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "InteractiveUtils", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "PrecompileTools", "Preferences", "Printf", "Random", "SparseArrays"]
git-tree-sha1 = "459de98eedc27ea3b19bd1924c11881569cfb834"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.13.55"
weakdeps = ["BFloat16s", "ChainRulesCore", "GPUArraysCore", "LogExpFunctions", "SpecialFunctions", "StaticArrays"]

    [deps.Enzyme.extensions]
    EnzymeBFloat16sExt = "BFloat16s"
    EnzymeChainRulesCoreExt = "ChainRulesCore"
    EnzymeGPUArraysCoreExt = "GPUArraysCore"
    EnzymeLogExpFunctionsExt = "LogExpFunctions"
    EnzymeSpecialFunctionsExt = "SpecialFunctions"
    EnzymeStaticArraysExt = "StaticArrays"

[[deps.EnzymeCore]]
git-tree-sha1 = "8272a687bca7b5c601c0c24fc0c71bff10aafdfd"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.12"
weakdeps = ["Adapt"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"

[[deps.Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "49dfd66929c794a6ec806bcb48d4d5d4b1280d11"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.184+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "797762812ed063b9b94f6cc7742bc8883bb5e69e"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.9.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "b66970a70db13f45b7e57fbda1736e1cf72174ea"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Flux]]
deps = ["Adapt", "ChainRulesCore", "Compat", "EnzymeCore", "Functors", "LinearAlgebra", "MLCore", "MLDataDevices", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "Setfield", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "2c35003ec8dafabdc48549102208b1b15552cb33"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.16.4"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxCUDAcuDNNExt = ["CUDA", "cuDNN"]
    FluxEnzymeExt = "Enzyme"
    FluxMPIExt = "MPI"
    FluxMPINCCLExt = ["CUDA", "MPI", "NCCL"]

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "910febccb28d493032495b7009dce7d7f7aee554"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.0.1"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "60a0339f28a233601cb74468032b5c302d5067de"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "Statistics"]
git-tree-sha1 = "be941842a40b6daac98496994ea69054ba4c5144"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.2.3"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "Tracy", "UUIDs"]
git-tree-sha1 = "eb1e212e12cc058fa16712082d44be499d23638c"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.6.1"

[[deps.GPUToolbox]]
git-tree-sha1 = "15d8b0f5a6dca9bf8c02eeaf6687660dafa638d0"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "0.2.0"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "e856eef26cf5bf2b0f95f8f4fc37553c72c8641c"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.2"

    [deps.HDF5.extensions]
    MPIExt = "MPI"

    [deps.HDF5.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "e94f84da9af7ce9c6be049e9067e511e17ff89ec"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.6+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Healpix]]
deps = ["CFITSIO", "LazyArtifacts", "Libsharp", "LinearAlgebra", "Pkg", "Printf", "Random", "RecipesBase", "StaticArrays", "Test"]
git-tree-sha1 = "611bfa6aa0a90e6f8188ddbb3de1a4b9cf18ca6e"
uuid = "9f4e344d-96bc-545a-84a3-ae6b9e1b672b"
version = "4.2.4"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "92f65c4d78ce8cdbb6b68daf88889950b0a99d11"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.12.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "57e9ce6cf68d0abf5cb6b3b4abf9bedf05c939c0"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.15"

[[deps.InfiniteArrays]]
deps = ["ArrayLayouts", "FillArrays", "Infinities", "LazyArrays", "LinearAlgebra"]
git-tree-sha1 = "e61675cbcf3ce57ea3566b0abcfe46e4a521bf6f"
uuid = "4858937d-0d70-526a-a4dd-2d5cb5dd786c"
version = "0.12.15"
weakdeps = ["Statistics"]

    [deps.InfiniteArrays.extensions]
    InfiniteArraysStatisticsExt = "Statistics"

[[deps.Infinities]]
git-tree-sha1 = "437cd9b81b649574582507d8b9ca3ffc981f2fc2"
uuid = "e1ba4f0e-776d-440f-acd9-e1d2e9742647"
version = "0.1.11"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "8594fac023c5ce1ef78260f24d1ad18b4327b420"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.4"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "0f14a5456bdc6b9731a5682f439a672750a09e48"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.0.4+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues", "TranscodingStreams"]
git-tree-sha1 = "d97791feefda45729613fafeccc4fbef3f539151"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.5.15"
weakdeps = ["UnPack"]

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JMcDM]]
deps = ["Requires"]
git-tree-sha1 = "e26d5db41aa1b96d4ed23b46eeeca34116214661"
uuid = "358108f5-d052-4d0a-8344-d5384e00c0e5"
version = "0.7.24"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "6ac9e4acc417a5b534ace12690bc6973c25b862f"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.3"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.Kaleido_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43032da5832754f58d14a91ffbe86d5f176acda9"
uuid = "f7e6163d-2fa5-5f23-b69c-1db539e41963"
version = "0.2.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "4efa9cec6f308e0f492ea635421638bff81cf6f8"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.36"
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "9c7c721cfd800d87d48c745d8bfb65144f0a91df"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.4.2"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "2ea068aac1e7f0337d381b0eae3110581e3f3216"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.37+2"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "MatrixFactorizations", "SparseArrays"]
git-tree-sha1 = "35079a6a869eecace778bcda8641f9a54ca3a828"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "1.10.0"
weakdeps = ["StaticArrays"]

    [deps.LazyArrays.extensions]
    LazyArraysStaticArraysExt = "StaticArrays"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.LibTracyClient_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d2bc4e1034b2d43076b50f0e34ea094c2cb0a717"
uuid = "ad6e5548-8b26-5c9f-8ef3-ef0ad883f3a5"
version = "0.9.1+6"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libsharp]]
deps = ["Libdl", "libsharp2_jll"]
git-tree-sha1 = "e09051a3f95b83091fc9b7a26e6c585c6691b5bc"
uuid = "ac8d63fe-4615-43ae-9860-9cd4a3820532"
version = "0.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoweredCodeUtils]]
deps = ["Compiler", "JuliaInterpreter"]
git-tree-sha1 = "16f11159553e5869972d7ca6a8b861e37197b1f8"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.4.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLDataDevices]]
deps = ["Adapt", "Compat", "Functors", "Preferences", "Random"]
git-tree-sha1 = "1b351cccb58e366504a5135d10e019234a27d792"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.10.1"

    [deps.MLDataDevices.extensions]
    MLDataDevicesAMDGPUExt = "AMDGPU"
    MLDataDevicesCUDAExt = "CUDA"
    MLDataDevicesChainRulesCoreExt = "ChainRulesCore"
    MLDataDevicesChainRulesExt = "ChainRules"
    MLDataDevicesComponentArraysExt = "ComponentArrays"
    MLDataDevicesFillArraysExt = "FillArrays"
    MLDataDevicesGPUArraysExt = "GPUArrays"
    MLDataDevicesMLUtilsExt = "MLUtils"
    MLDataDevicesMetalExt = ["GPUArrays", "Metal"]
    MLDataDevicesOneHotArraysExt = "OneHotArrays"
    MLDataDevicesReactantExt = "Reactant"
    MLDataDevicesRecursiveArrayToolsExt = "RecursiveArrayTools"
    MLDataDevicesReverseDiffExt = "ReverseDiff"
    MLDataDevicesSparseArraysExt = "SparseArrays"
    MLDataDevicesTrackerExt = "Tracker"
    MLDataDevicesZygoteExt = "Zygote"
    MLDataDevicescuDNNExt = ["CUDA", "cuDNN"]
    MLDataDevicesoneAPIExt = ["GPUArrays", "oneAPI"]

    [deps.MLDataDevices.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "a772d8d1987433538a5c226f79393324b55f7846"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.8"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "d72d0ecc3f76998aac04e446547259b9ae4c265f"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.3.1+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "e214f2a20bdd64c04cd3e4ff62d3c9be7e969a59"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.4+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MatrixFactorizations]]
deps = ["ArrayLayouts", "LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "6731e0574fa5ee21c02733e397beb133df90de35"
uuid = "a3b82374-2e81-5b9e-98ce-41277c0e4c87"
version = "2.2.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Metaheuristics]]
deps = ["Distances", "JMcDM", "LinearAlgebra", "Pkg", "Printf", "Random", "Reexport", "Requires", "SearchSpaces", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "03d457e957cb8238d787525b8fec04ece4cbac9a"
uuid = "bcdb8e00-2c21-11e9-3065-2b553b22f898"
version = "3.4.0"

[[deps.Metalhead]]
deps = ["Artifacts", "BSON", "ChainRulesCore", "Flux", "Functors", "JLD2", "LazyArtifacts", "MLUtils", "NNlib", "PartialFunctions", "Random", "Statistics"]
git-tree-sha1 = "7d3cdd8acb8ccdf82bb80d07f33f020b0976ddc5"
uuid = "dbeba491-748d-5e0e-a39e-b530a07fa0cc"
version = "0.9.5"
weakdeps = ["CUDA"]

    [deps.Metalhead.extensions]
    MetalheadCUDAExt = "CUDA"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "4abc63cdd8dd9dd925d8e879cda280bedc8013ca"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.30"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NVTX]]
deps = ["Colors", "JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "1a24c3430fa2ef3317c4c97fa7e431ef45793bd2"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "1.0.0"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cd475b587ff77910789a18e68da789fc446a2a05"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.2.1+0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.ObjectFile]]
deps = ["Reexport", "StructIO"]
git-tree-sha1 = "09b1fe6ff16e6587fa240c165347474322e77cf1"
uuid = "d8793406-e978-5875-9003-1fc021f44a92"
version = "0.4.4"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "bfe8e84c71972f77e775f75e6d8048ad3fdbe8bc"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.10"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "ec764453819f802fc1e144bfe750c454181bd66d"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.8+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "87510f7292a2b21aeff97912b0898f9553cc5c2c"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "131dc319e7c58317e8c6d5170440f6bdaee0a959"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.6"

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"

    [deps.Optimisers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.ParameterSchedulers]]
deps = ["InfiniteArrays", "Optimisers"]
git-tree-sha1 = "c62f0da0663704d0472ae578c9bb802c44e70a4c"
uuid = "d7d3b36b-41b8-4d0d-a2bf-768c6151755e"
version = "0.4.3"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.PartialFunctions]]
deps = ["MacroTools"]
git-tree-sha1 = "ba0ea009d9f1e38162d016ca54627314b6d8aac8"
uuid = "570af359-4316-4cb7-8c74-252c00c2016b"
version = "1.2.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "28278bb0053da0fd73537be94afd1682cc5a0a83"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.21"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlotlyKaleido]]
deps = ["Artifacts", "Base64", "JSON", "Kaleido_jll"]
git-tree-sha1 = "9ef5c9e588ec7e912f01a76c7fd3dddf1913d4f2"
uuid = "f2990250-8cf9-495f-b13a-cce12b45703c"
version = "2.3.0"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "4fb7c9595eaad32d817cac8c5fa1f90daa83aa4c"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.3"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "2b2127e64c1221b8204afe4eb71662b641f33b82"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.66"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "972089912ba299fba87671b025cd0da74f5f54f7"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.0"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "d95ed0324b0799843ac6f7a6a85e65fe4e5173f0"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.5"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "13c5103482a8ed1536a54c08d0e742ae3dca2d42"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.4"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"
weakdeps = ["Enzyme"]

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "f6f7d30fb0d61c64d0cfe56cf085a7c9e7d5bc80"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.8.0"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "1147f140b4c8ddab224c94efa9569fc23d63ab44"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.3.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SearchSpaces]]
deps = ["Combinatorics", "Random"]
git-tree-sha1 = "2662fd537048fb12ff34fabb5249bf50e06f445b"
uuid = "eb7571c6-2196-4f03-99b8-52a5a35b3163"
version = "0.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpatialIndexing]]
git-tree-sha1 = "84efe17c77e1f2156a7a0d8a7c163c1e1c7bdaed"
uuid = "d4ead438-fe20-5cc5-a293-4fd39a41b74c"
version = "0.1.6"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0feb6b9031bd5c51f9072393eb5ab3efd31bf9e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.13"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "b81c5035922cc89c2d9523afc6c54be512411466"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.5"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "8e45cecc66f3b42633b8ce14d431e8e57a3e242e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "8ad2e38cbb812e29348719cc63580ec1dfeb9de4"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.1"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructIO]]
git-tree-sha1 = "c581be48ae1cbf83e899b14c07a807e1787512cc"
uuid = "53d494c1-5632-5724-8f4c-31dff12d585f"
version = "0.3.1"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TSne]]
deps = ["Distances", "LinearAlgebra", "Printf", "ProgressMeter", "Statistics"]
git-tree-sha1 = "6f1dfbf9dad6958439816fa9c5fa20898203fdf4"
uuid = "24678dba-d5e9-5843-a4c6-250288b04835"
version = "1.3.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "91dbaee0f50faa4357f7e9fc69442c7b6364dfe5"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.5"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a29cbf3968d36022198bcc6f23fdfd70f7caf737"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.7.10"

    [deps.Zygote.extensions]
    ZygoteAtomExt = "Atom"
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Atom = "c52e3926-4ff0-5f6e-af25-54175e0327b1"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "434b3de333c75fc446aa0d19fc394edafd07ab08"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.7"

[[deps.cuDNN]]
deps = ["CEnum", "CUDA", "CUDA_Runtime_Discovery", "CUDNN_jll"]
git-tree-sha1 = "8c20cd99f74552772a26712578db42a8a4200cec"
uuid = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
version = "1.4.3"

[[deps.demumble_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6498e3581023f8e530f34760d18f75a69e3a4ea8"
uuid = "1e29f10c-031c-5a83-9565-69cddfc27673"
version = "1.3.0+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f5733a5a9047722470b95a81e1b172383971105c"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.3+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libsharp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl", "Pkg"]
git-tree-sha1 = "1029b8ace2e7cb7c6bafde9300380acd6faee75a"
uuid = "180207a7-b08e-5162-af94-7d62a04fe081"
version = "1.0.2+2"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d5a767a3bb77135a99e433afe0eb14cd7f6914c3"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═9e8a8120-a2d0-4898-b5f4-bd69b722fd39
# ╠═974e1d57-1cc7-4258-8ff1-f79ca55e7c9f
# ╠═36d58800-cd47-4a74-aec6-b61306ec0136
# ╠═19c9e110-e09a-4ec1-8aaf-0507236912e5
# ╠═c48e61dd-12e7-4eb8-aaf0-9036d7890b5e
# ╠═9cf663af-d59a-4c2b-956e-038c457fd5d2
# ╠═7629650d-95eb-48ff-bc86-fd9e2249d1ea
# ╟─b46bd074-ad00-4cc4-9967-b1952f4931d3
# ╠═62655df3-0e96-4567-b882-9b6babc972f6
# ╠═eb429d30-e5ad-4d27-9cbb-7c0ad2315948
# ╠═1be70e89-bf9d-4702-be40-8ed119d257ec
# ╠═2a36b358-9fd6-4d06-8078-c467a6b070a3
# ╠═2cd412fd-c57a-4caf-b73f-ac926ea47f8c
# ╠═1ccc0ec1-798b-42cb-a91c-5f6b06820ca7
# ╠═235ec0aa-5ef6-47ce-85b2-b96407266cd3
# ╠═5abc5ef3-93a0-4eef-b6bf-7e1ddf16c232
# ╠═f5eb2333-be02-4891-9bde-c5bc917278d4
# ╠═c5f7712f-d35a-49ee-a02b-4a96b04cbae3
# ╠═988f1f0e-e6f9-43a2-998f-e8373185b26f
# ╠═76d98574-1ca1-4652-ac52-fa3e4d5f4608
# ╟─676642c4-8205-4523-b069-787be1040a5e
# ╠═1e85c1c8-ed8d-44b9-8d3e-025b5014b97b
# ╠═2fd9a503-cf69-4c80-8af1-ac68a7a18886
# ╠═58eaa0a2-1289-4629-bad4-2cdedde0dfc2
# ╠═f4df92a7-1501-41c7-bda5-777ffeefe488
# ╠═818f312f-2370-4606-bc92-579030e9cb97
# ╠═b2d8ae0d-c4c8-46fa-beba-eeceed8e1321
# ╠═68bcf7d0-4163-45f4-ab67-3ef6ca8c57db
# ╠═143fe031-187f-41b8-95ae-f1eee558ca31
# ╠═9d2450b7-e96e-48f9-9100-9b862d0074d5
# ╠═b695c9d5-cf22-44eb-a877-0701a88f5f1d
# ╠═2b8e3682-10e6-426a-8c1d-ae774c270844
# ╠═e2df0cc3-f202-4366-aa7c-bd18046151e6
# ╠═fdb76d1a-e114-4126-a214-cf9d00a80c09
# ╠═d622fc67-7f2e-4459-8e30-aac42a312ad3
# ╠═11ed37bf-aa3c-4bb4-82ce-0a0961247e4c
# ╠═7771d130-0a8a-4803-8a6b-966533c887f1
# ╠═4f7193f0-b92d-45dc-a449-7fe189eb0ca1
# ╠═927200b1-d4fc-4004-a468-f599729e178c
# ╠═7e525230-9c74-442c-987d-a1cabec96b7a
# ╠═6382acfb-f9de-4ff3-975c-ca2abf724493
# ╟─45b2a58f-fbd6-44ee-aa3c-005a45086c54
# ╠═5adec3e2-acd4-4895-bb4e-d5ebf7bde0c2
# ╠═def83d24-1dea-4528-b85b-fc25fb43abc1
# ╠═0fc67113-d195-4ceb-894c-f633d4c24b0c
# ╠═2cb046b7-4bec-4bed-bdaa-a3cc260574c7
# ╠═b5703c40-e188-4a74-abb2-c1a9b1a19079
# ╠═4483c965-6782-4195-ba92-29beb9a7c635
# ╠═b953dca6-8d64-473b-9d86-705a5ee443ee
# ╠═3ca58bfb-8b77-4354-b2e3-e6decf3f92e9
# ╠═584ae1ac-7d69-4e17-b2bd-30736d52bb90
# ╠═a8cea190-fdd2-496e-83d5-b186a1adbecd
# ╠═229949e4-aae3-4c00-931f-382795f38b13
# ╠═4bea2566-9b11-440b-959d-f131c79dc47e
# ╠═c04c2211-a87e-41ee-a42a-e2ea03c9d0bc
# ╠═b88703d2-72ed-41be-a799-732166740855
# ╠═3e6d4563-1e35-4adb-84cb-2cb5c6eacea4
# ╠═86c84c4b-6449-4a7c-9ae8-6408ded55b70
# ╠═d6088ec8-43a5-42ee-b3e6-e890c8a3de4f
# ╠═f10fbe7f-e161-4d73-9e27-56d32860b42d
# ╠═f82e40a8-1321-4ffe-982e-e72aacdae538
# ╠═948e21ce-b897-4f69-adcd-ed176761798e
# ╠═7513d195-e637-4e68-bdfa-7d879c5ca326
# ╠═cb7c950d-40dc-4890-9e8f-20b8b77448cd
# ╠═c9b8fae4-7199-4e4d-a8ee-6bca22770bb2
# ╟─2033dffb-7e8c-4f27-ac4a-c0c9522eb5c8
# ╠═0b4f54c4-0a84-466c-9c19-4cfe68e82a98
# ╟─8339c5a0-6fad-451b-b4df-ed22588d806a
# ╠═7f2a30cc-eaab-49e7-ba6b-145d4e00f95c
# ╠═af479841-ac07-4b8d-b58d-7575a94f9be4
# ╠═a59c278e-8779-48d0-9c82-4cf2aad0e471
# ╠═5732782d-c8f5-42d2-a94b-dfcc5cba0a80
# ╠═41370b84-a791-4e5f-810c-d5ae458679a1
# ╠═aa9d5025-ad7f-4f2a-9c41-8cf1fe72db90
# ╠═818c44ef-86bf-410d-b698-8cff6c6969b1
# ╠═d4034569-cd45-4a98-94d2-0de39491201e
# ╠═25360328-f93a-4ab4-93c9-73c5808a8847
# ╠═c3dce92a-46dc-4af8-816b-c1880858b85b
# ╟─63b4fb92-783f-445e-919e-b3a3519dd06a
# ╠═98863e40-d3d1-4eb6-9325-67373f3f973e
# ╠═8e5e4a99-35c5-426a-8440-bf773be6f252
# ╟─36f93d83-45b3-45f3-a881-ac20f8cb2c6e
# ╠═fb981c7e-24bc-4983-abc1-182989122562
# ╠═da15d3eb-e776-43f2-8026-a06b61c33965
# ╠═4501fdf1-42d6-4703-bcf1-3be915b2931b
# ╠═8d501f0b-f45c-4e26-9de8-fa1ff0eb2b2f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
