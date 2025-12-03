### A Pluto.jl notebook ###
# v0.20.9

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

# ╔═╡ 0d6e4ef8-bbcf-444b-9649-9d75944585c1
using HDF5, PlutoUI, DSP, MLUtils, PlutoPlotly, PlutoLinks, PlutoHooks, StatsBase, LinearAlgebra

# ╔═╡ 7e457cff-4f26-47f8-a22d-12c4e563083a
using Random

# ╔═╡ a34941e8-32ea-445f-8300-c9a9e3632efe
using Healpix

# ╔═╡ 49d2c89e-b634-44c7-b0c6-aba330a8c6d9
plots = @ingredients("Plotting.jl")

# ╔═╡ 48f55ae6-aff5-4287-8c62-bc83b98ae50f
@__DIR__

# ╔═╡ 3dbf0176-2c64-11ef-01a3-e77e62dbf93b
begin
	eqfiles = [
        # h5open(joinpath(@__DIR__, "new_20_syn_P.hdf5"), "r"),
        # h5open(joinpath(@__DIR__, "pbseq.hdf5"), "r"),
		h5open(joinpath("/home/pawan/Notebooks/","deep_EQS_P_1.hdf5"), "r"),
        # h5open(joinpath("/mnt/NAS/EQData/Deep-Earthquakes/","deep_EQS_pP.hdf5"), "r"),
    ]
    eqfilesS = [
        # h5open(joinpath(@__DIR__, "new_20_syn_P.hdf5"), "r"),
        # h5open(joinpath(@__DIR__, "pbseq.hdf5"), "r"),
		# h5open(joinpath("/home/pawan/Notebooks/","deep_EQS_P_1.hdf5"), "r"),
        h5open(joinpath("/mnt/NAS/EQData/Deep-Earthquakes/","deep_EQS_pP.hdf5"), "r"),
    ]
    # eqfilesS=[h5open("S_dataset_40eq.hdf5", "r")
    # ]
    # h5open("1.0_Preprocessed_Data/eq_all_P.hdf5", "r"),h5open("1.0_Preprocessed_Data/japan_finiteP.hdf5", "r"),h5open("1.0_Preprocessed_Data/pbseq.hdf5", "r"),h5open("1.0_Preprocessed_Data/hindukush_finiteP.hdf5", "r"),h5open("1.0_Preprocessed_Data/fiji_finiteP.hdf5", "r"),h5open("1.0_Preprocessed_Data/sumatra_finite.hdf5", "r"),h5open("1.0_Preprocessed_Data/eq_pb2.hdf5", "r")];
end

# ╔═╡ 59e929f0-9403-4ba7-baa9-7d603c07fd9f
nr_min_pixel = 20

# ╔═╡ b6d2d61e-f8be-41c4-a7ad-dfd688846120
ntau_min = 20

# ╔═╡ 1be08651-2356-4aba-8f3d-a1ee72835d1f
window = tukey(400, 1)

# ╔═╡ b400589b-bbf9-4126-a9a4-434d7465a60e
plot(window)

# ╔═╡ 818a2774-9cf4-4e9e-98fe-c258cb6dd451
function taper(x, alpha=0.1)
    return cat(tukey(size(x, 1), alpha), dims = ndims(x)) .* x
    # return cat(
    #     (1.0 .- tukey(size(x, 1) - 300, 0.5, padding = 300, zerophase = true)),
    #     dims = ndims(x),
    # ) .* x
end

# ╔═╡ 9061c885-5f5a-4279-8140-980a2958f08f
plot(taper(randn(128)))

# ╔═╡ ce1b1918-8e01-4fed-aa4b-dc216ce832df
begin  ## S pixel wise
    SDeq_train_pixel = []
    SDeq_train_stn_pixel = []
    SDeq_test_stn_pixel = []
    SDeq_test_pixel = []
    SDeqnames_pixel = []
    for eqfile in eqfilesS
        map(keys(eqfile)) do eq
            map(keys(eqfile[eq])) do pixel
                stn_names = keys(eqfile[eq][pixel])
                data = mapreduce(hcat, stn_names) do stn
                    return Float32.(Array(eqfile[eq][pixel][stn])[:])
                end

                # minimum number of stations 
                if (size(data, 2) > nr_min_pixel)
					d = data
                    # d = taper(d)
					# d = resample(d, 0.5, dims=1)
					d = d[101:300, :]
					d = Float32.(d)
                    # d = normalise(d, dims = 1)
					d = taper(d, 1)
					d = normalise(d, dims = 1)
					Random.seed!(1234)
                    D1, D2 = splitobs((d, stn_names), at = (0.9), shuffle = true)
                    push!(SDeq_train_pixel, getobs(D1[1]))
                    push!(SDeq_train_stn_pixel, getobs(D1[2]))
                    push!(SDeq_test_pixel, getobs(D2[1]))
                    push!(SDeq_test_stn_pixel, getobs(D2[2]))
                    push!(SDeqnames_pixel, string(eq, "_", pixel))
                end
            end
        end
    end
end

# ╔═╡ 177beb36-b0a2-4a2e-90f4-c097e42a046c
begin  ## pixel wise
    Deq_train_pixel = []
    Deq_train_stn_pixel = []
    Deq_test_stn_pixel = []
    Deq_test_pixel = []
    Deqnames_pixel = []
    for eqfile in eqfiles
        map(keys(eqfile)) do eq
            map(keys(eqfile[eq])) do pixel
                stn_names = keys(eqfile[eq][pixel])
                data = mapreduce(hcat, stn_names) do stn
                    return Float32.(Array(eqfile[eq][pixel][stn])[1:400])
                end

                # minimum number of stations 
                if (size(data, 2) > nr_min_pixel)
					d = data
                    d = taper(d, 0.1)
					d = d[101:300, :]
					d = Float32.(d)
                    # d = normalise(d, dims = 1)
					d = taper(d, 0.1)
					d = normalise(d, dims = 1)
					Random.seed!(1234)
                    D1, D2 = splitobs((d, stn_names), at = (0.9), shuffle = true)
                    push!(Deq_train_pixel, getobs(D1[1]))
                    push!(Deq_train_stn_pixel, getobs(D1[2]))
                    push!(Deq_test_pixel, getobs(D2[1]))
                    push!(Deq_test_stn_pixel, getobs(D2[2]))
                    push!(Deqnames_pixel, string(eq, "_", pixel))
                end
            end
        end
    end
end

# ╔═╡ 7832dbde-9c46-41f1-b216-52348a728bf0
# ╠═╡ disabled = true
#=╠═╡
begin # one state for each earthquake
    Deq_train = []
    Deq_train_stn = []
    Deq_test_stn = []
    Deq_test = []
    Deqnames = []
    for eqfile in eqfiles
        map(keys(eqfile)) do eq
            eqdata=map(keys(eqfile[eq])) do pixel
                stn_names = keys(eqfile[eq][pixel])
                data = mapreduce(hcat, stn_names) do stn
                    return Float32.(Array(eqfile[eq][pixel][stn])[1:400])
                end

                # minimum number of stations 
            
					d = data
                    d = taper(d)
					d = d[101:300, :]
					d = Float32.(d)
                    d = normalise(d, dims = 1)
					d = taper(d)
                   return (d, stn_names)
                end
			
			D = cat(first.(eqdata)..., dims=2)
			stn_names = vcat(last.(eqdata)...)
			
			 D1, D2 = splitobs((D, stn_names), at = (0.9), shuffle = true)
                    push!(Deq_train, getobs(D1[1]))
                    push!(Deq_train_stn, getobs(D1[2]))
                    push!(Deq_test, getobs(D2[1]))
                    push!(Deq_test_stn, getobs(D2[2]))
                    push!(Deqnames, string(eq))
        
    end

end
end
  ╠═╡ =#

# ╔═╡ d071b103-1293-4a07-8f82-5b5436148270
# plot(mean(envelope((Sdata_pixel.Dtrain[findalleq("okt5_14", Sdata_pixel)])), dims=2))

# ╔═╡ eae96983-5a43-4b4f-82f0-aeccd6b57ac5
function findalleq(eq, data)
	first(findall(x -> occursin(eq, x), data.Dnames))
end

# ╔═╡ e5eaf9b7-bda0-4987-9043-affee18c9268
envelope(x) = abs.(hilbert(x))

# ╔═╡ aeb47fa8-d125-49a6-b92a-7051d3e61fd4


# ╔═╡ f45cd343-86e2-446d-822d-5f66d31067fd
data_pixel_all = (;
    Dtrain = vcat(Deq_train_pixel, SDeq_train_pixel),
    Dtest = vcat(Deq_test_pixel, SDeq_test_pixel),
    Deq_train_stn = vcat(Deq_train_stn_pixel, SDeq_train_stn_pixel),
    Deq_test_stn = vcat(Deq_test_stn_pixel, SDeq_test_stn_pixel),
    Dnames = vcat(Deqnames_pixel, SDeqnames_pixel),
);

# ╔═╡ e2618849-145d-41ad-a335-09aebb9dead9
data_pixel_pP = (;
    Dtrain = SDeq_train_pixel,
    Dtest = SDeq_test_pixel,
    Deq_train_stn = SDeq_train_stn_pixel,
    Deq_test_stn = SDeq_test_stn_pixel,
    Dnames = SDeqnames_pixel,
);

# ╔═╡ 41b82236-fd5b-441f-aa92-3a2852582e63
#=╠═╡
data = (;
    Dtrain = Deq_train,
    Dtest = Deq_test,
    Deq_train_stn = Deq_train_stn,
    Deq_test_stn = Deq_test_stn,
    Dnames = Deqnames,
);
  ╠═╡ =#

# ╔═╡ 2e720de5-339d-40aa-a30a-67f39c36f1e8
#=╠═╡
plot(
    scatter(y = size.(data.Dtrain, 2), x = Deqnames),
    Layout(width = 700, height = 400, title = "Number Of Seismograms Per Earthquake"),
)
  ╠═╡ =#

# ╔═╡ 34365606-c3a5-4196-8f0f-24a365584d54
md"""
$(@bind resample_eq_data Button("Resample Eq"))
$(@bind resample_eq_data_name TextField())
"""

# ╔═╡ 95cdadbd-6332-4421-9305-5504f9ac2bc7
200 - 121

# ╔═╡ c05ffb26-f115-4cd1-bd10-d0cc3f09b519
tgrid = range(-100, 100, length = 200)

# ╔═╡ b2e9580b-94de-4218-8212-275075f61274
step(tgrid)

# ╔═╡ 1d165af4-0389-4433-8251-df92bf6466dc
#=╠═╡
plot(data.Dtrain[1][:, 1])
  ╠═╡ =#

# ╔═╡ bcb75fab-56ec-440b-8cbb-c47f9567f96d
#=╠═╡
Deqnames
  ╠═╡ =#

# ╔═╡ f1ae8513-a969-4ca9-962e-634c4c806acc
function bandpass_filter(signal, fs, f_peak, bandwidth)
    # Design a band-pass filter around the peak frequency
    low_cutoff = f_peak - bandwidth / 2
    high_cutoff = f_peak + bandwidth / 2
    bp_filter = Butterworth(2, Bandpass(low_cutoff / (fs / 2), high_cutoff / (fs / 2)))
    return filtfilt(bp_filter, signal)
end

# ╔═╡ 55fa5379-00cb-4895-9fc2-9759193200f2
function correct_polarity!(signals, reference) 
    for i in 1:size(signals, 2)
        signal = view(signals, :, i)
		nt = length(signal)
        # Compute the dot product with the reference signal
        # dot_product = dot(signal, reference)
		correlation = crosscor(signal, reference, -nt+1:nt-1)
		I = argmax(abs.(correlation))
        # Reverse polarity in-place if the dot product is negative
        if correlation[I] < 0
			rmul!(signal, -1.0)
        end
    end
end

# ╔═╡ 0c0d60ee-759b-476f-abf7-c9b2a540782f
function correct_polarity_D!(D, Dnames, Dref=D)
	eqnames = map(x->join(split(x, "_")[1:end-1], "_"), Dnames)
	map(eqnames) do eq
		kidx = findall(x -> occursin(eq, x), Dnames)
		reference = mean(map(kidx) do k
			mean(Dref[k], dims=2)
		end)
		map(kidx) do k
			D1 = D[k] 
			correct_polarity!(D1, reference)
		end
	end
	return D
end

# ╔═╡ 23d7627a-c041-432e-91b0-4c66987c5ce3
begin
	data_pixel = (;
	    Dtrain = Deq_train_pixel,
	    Dtest = Deq_test_pixel,
	    Deq_train_stn = Deq_train_stn_pixel,
	    Deq_test_stn = Deq_test_stn_pixel,
	    Dnames = Deqnames_pixel,
	);
	# data_pixel = (;
	#     Dtrain = Deq_train_pixel,
	#     Dtest = Deq_test_pixel,
	#     Deq_train_stn = Deq_train_stn_pixel,
	#     Deq_test_stn = Deq_test_stn_pixel,
	#     Dnames = Deqnames_pixel,
	# );
	correct_polarity_D!(data_pixel.Dtrain, data_pixel.Dnames)
	correct_polarity_D!(data_pixel.Dtest, data_pixel.Dnames, data_pixel.Dtrain)
end;

# ╔═╡ daace6f5-5e9a-4aca-adb2-8f4c4acfe1ae
plot(mean(envelope((data_pixel.Dtrain[findalleq("20021117_045353_okt", data_pixel)])), dims=2))

# ╔═╡ b05e2464-ac21-4aa5-95a6-362ea05fc7ae
data_pixel.Dnames

# ╔═╡ 30c7da7b-3bbc-49bd-a505-23b4de93b5be
kidx_okt1 = (findall(x -> occursin("20130524_054448", x), data_pixel.Dnames))

# ╔═╡ b102c00f-9444-4aa2-91e0-2992da4a5f44
let
    resample_eq_data
    i = randobs(findall(x -> occursin(resample_eq_data_name, x), data_pixel.Dnames))
    eq, eqname = data_pixel.Dtrain[i], data_pixel.Dnames[i]
    eqstn = data_pixel.Deq_train_stn[i]
    dplot, dplotnames = randobs((eq, eqstn), 20)
	
    plots.plot_data(
        unstack(dplot, dims = 2),
        labels = dplotnames,
        title = "Sample Data: $(eqname)",
        opacitys = fill(1, 20),
		tgrid =tgrid
    )
end

# ╔═╡ 5e7754da-1bbd-4309-aa86-231f39b51e74
let
    resample_eq_data
    i = randobs(findall(x -> occursin(resample_eq_data_name, x), SDeqnames_pixel))
    eq, eqname = data_pixel_pP.Dtrain[i], data_pixel_pP.Dnames[i]
    eqstn = data_pixel_pP.Deq_train_stn[i]
    dplot, dplotnames = randobs((eq, eqstn), 3)
    plots.plot_data(
        unstack(dplot, dims = 2),
        labels = dplotnames,
        title = "Sample Data: $(eqname)",
        opacitys = [1, 1, 0.5, 0.5],
		tgrid =tgrid
    )
end

# ╔═╡ 2184693e-7a32-460d-951c-c79c4d0466a8
#=╠═╡
size(data.Dtrain[1])
  ╠═╡ =#

# ╔═╡ 98c005d2-046e-4bdc-b18c-840bc85dba9e
# eq_list = map(event_list["Name"], event_list["Code"]) do eq, code
#     list1 = eq, code
# end

# ╔═╡ c96154ad-f4f8-4329-8f2d-dd5e84e082da
# phase = [h5open("1.1_Phase_time.hdf5", "r")];

# ╔═╡ 391e5dce-6d81-454c-ad4a-202399aa02f5
# begin
#     pP = Array(phase[1]["fij1"]["pP"])
#     sP = Array(phase[1]["fij1"]["sP"])
# end

# ╔═╡ b27ade4d-96f1-4329-8721-2f25b82255b6
# event_list= CSV.File("events_list40.csv")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
Healpix = "9f4e344d-96bc-545a-84a3-ae6b9e1b672b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
DSP = "~0.7.9"
HDF5 = "~0.17.2"
Healpix = "~4.2.2"
MLUtils = "~0.4.4"
PlutoHooks = "~0.0.5"
PlutoLinks = "~0.1.6"
PlutoPlotly = "~0.4.6"
PlutoUI = "~0.7.59"
StatsBase = "~0.34.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.2"
manifest_format = "2.0"
project_hash = "190693e4d85a18bb0ef4433f892b7a97dcb6191c"

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
deps = ["CompositionsBase", "ConstructionBase", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown"]
git-tree-sha1 = "96bed9b1b57cf750cca50c311a197e306816a1cc"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.39"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsDatesExt = "Dates"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsTestExt = "Test"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "680b3b8759bd4c54052ada14e52355ab69e07876"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c3b238aa28c1bebd4b5ea4988bebf27e9a01b72b"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.0.1"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "e2144b631226d9eeab2d746ca8880b7ccff504ae"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.3"

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

[[deps.BaseDirs]]
git-tree-sha1 = "cb25e4b105cc927052c2314f8291854ea59bf70a"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.2.4"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.CFITSIO]]
deps = ["CFITSIO_jll"]
git-tree-sha1 = "fc0abb338eb8d90bc186ccf0a47c90825952c950"
uuid = "3b1b4be9-1499-4b22-8d78-7db3344d1961"
version = "1.4.2"

[[deps.CFITSIO_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b90d32054fc88f97dd926022f554180e744e4d7d"
uuid = "b3e40c51-02ae-5482-8a39-3ace5868dcf4"
version = "4.4.0+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

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

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

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
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

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

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "0df00546373af8eee1598fb4b2ba480b1ebe895c"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.10"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

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

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+3"

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

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
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
git-tree-sha1 = "82a471768b513dc39e471540fdadc84ff80ff997"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.3+3"

[[deps.Healpix]]
deps = ["CFITSIO", "LazyArtifacts", "Libsharp", "LinearAlgebra", "Pkg", "Printf", "Random", "RecipesBase", "StaticArrays", "Test"]
git-tree-sha1 = "0be8d5296b8dbb02b5808c01c2911cd019338492"
uuid = "9f4e344d-96bc-545a-84a3-ae6b9e1b672b"
version = "4.2.2"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "50aedf345a709ab75872f80a2779568dc0bb461b"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.11.2+3"

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

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

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

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "10da5154188682e5c0726823c2b5125957ec3778"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.38"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "b9a838cd3028785ac23822cded5126b3da394d1a"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.31"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

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
deps = ["JuliaInterpreter"]
git-tree-sha1 = "688d6d9e098109051ae33d126fcfc88c4ce4a021"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "b45738c2e3d0d402dffa32b2c1654759a2ac35a4"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.4"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "7715e65c47ba3941c502bffb7f266a41a7f54423"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.2.3+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "70e830dab5d0775183c99fc75e4c24c614ed7142"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.1+2"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

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
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "1177f161cda2083543b9967d7ca2a3e24e721e13"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.26"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "e25c1778a98e34219a00455d6e4384e017ea9762"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "4.1.6+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+3"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

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
deps = ["AbstractPlutoDingetjes", "BaseDirs", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "Reexport", "TOML"]
git-tree-sha1 = "1ae939782a5ce9a004484eab5416411c7190d3ce"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.4.6"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "adc25dbd4d13f148f3256b6d4743fe7e63a71c4a"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.12"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

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

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

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
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "b5e7c125687aa818de948ef8a0b9dc59f043342c"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.7.1"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

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

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"
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
git-tree-sha1 = "47091a0340a675c738b1304b58161f3b0839d454"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.10"
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
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

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
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

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

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

    [deps.UnsafeAtomics.weakdeps]
    LLVM = "929cbde3-209d-540e-8aea-75f648917ca0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "46bf7be2917b59b761247be3f317ddf75e50e997"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.2+2"

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
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═0d6e4ef8-bbcf-444b-9649-9d75944585c1
# ╠═7e457cff-4f26-47f8-a22d-12c4e563083a
# ╠═a34941e8-32ea-445f-8300-c9a9e3632efe
# ╠═49d2c89e-b634-44c7-b0c6-aba330a8c6d9
# ╠═48f55ae6-aff5-4287-8c62-bc83b98ae50f
# ╠═3dbf0176-2c64-11ef-01a3-e77e62dbf93b
# ╠═59e929f0-9403-4ba7-baa9-7d603c07fd9f
# ╠═b6d2d61e-f8be-41c4-a7ad-dfd688846120
# ╠═1be08651-2356-4aba-8f3d-a1ee72835d1f
# ╠═b400589b-bbf9-4126-a9a4-434d7465a60e
# ╠═818a2774-9cf4-4e9e-98fe-c258cb6dd451
# ╠═9061c885-5f5a-4279-8140-980a2958f08f
# ╠═ce1b1918-8e01-4fed-aa4b-dc216ce832df
# ╠═177beb36-b0a2-4a2e-90f4-c097e42a046c
# ╠═7832dbde-9c46-41f1-b216-52348a728bf0
# ╠═daace6f5-5e9a-4aca-adb2-8f4c4acfe1ae
# ╠═d071b103-1293-4a07-8f82-5b5436148270
# ╠═eae96983-5a43-4b4f-82f0-aeccd6b57ac5
# ╠═e5eaf9b7-bda0-4987-9043-affee18c9268
# ╠═b05e2464-ac21-4aa5-95a6-362ea05fc7ae
# ╠═aeb47fa8-d125-49a6-b92a-7051d3e61fd4
# ╠═23d7627a-c041-432e-91b0-4c66987c5ce3
# ╠═f45cd343-86e2-446d-822d-5f66d31067fd
# ╠═e2618849-145d-41ad-a335-09aebb9dead9
# ╠═41b82236-fd5b-441f-aa92-3a2852582e63
# ╠═2e720de5-339d-40aa-a30a-67f39c36f1e8
# ╠═34365606-c3a5-4196-8f0f-24a365584d54
# ╠═95cdadbd-6332-4421-9305-5504f9ac2bc7
# ╠═c05ffb26-f115-4cd1-bd10-d0cc3f09b519
# ╠═b2e9580b-94de-4218-8212-275075f61274
# ╠═1d165af4-0389-4433-8251-df92bf6466dc
# ╠═bcb75fab-56ec-440b-8cbb-c47f9567f96d
# ╠═f1ae8513-a969-4ca9-962e-634c4c806acc
# ╠═0c0d60ee-759b-476f-abf7-c9b2a540782f
# ╠═55fa5379-00cb-4895-9fc2-9759193200f2
# ╠═30c7da7b-3bbc-49bd-a505-23b4de93b5be
# ╠═b102c00f-9444-4aa2-91e0-2992da4a5f44
# ╠═5e7754da-1bbd-4309-aa86-231f39b51e74
# ╠═2184693e-7a32-460d-951c-c79c4d0466a8
# ╠═98c005d2-046e-4bdc-b18c-840bc85dba9e
# ╠═c96154ad-f4f8-4329-8f2d-dd5e84e082da
# ╠═391e5dce-6d81-454c-ad4a-202399aa02f5
# ╠═b27ade4d-96f1-4329-8721-2f25b82255b6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
