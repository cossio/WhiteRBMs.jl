var documenterSearchIndex = {"docs":
[{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"EditURL = \"https://github.com/cossio/WhitenedRBMs.jl/blob/master/docs/src/literate/MNIST_center.jl\"","category":"page"},{"location":"literate/MNIST_center/#MNIST","page":"MNIST centered","title":"MNIST","text":"","category":"section"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"We begin by importing the required packages. We load MNIST via the MLDatasets.jl package.","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"Some references","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"https://www.jmlr.org/beta/papers/v17/14-237.html,","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"http://www.cs.toronto.edu/~tang/papers/RbmZM.pdf,","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"https://doi.org/10.1007/978-3-642-35289-8_3","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"import CairoMakie\nimport Makie\nimport MLDatasets\nimport Flux\nimport RestrictedBoltzmannMachines as RBMs\nimport WhitenedRBMs as WhiteRBMs\nusing Statistics: mean, var, std\nusing Random: bitrand\nusing ValueHistories: MVHistory\nusing RestrictedBoltzmannMachines: BinaryRBM\nusing WhitenedRBMs: whiten, blacken, whiten_visible_from_data\nnothing #hide","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"Useful function to plot MNIST digits.","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"\"\"\"\n    imggrid(A)\n\nGiven a four dimensional tensor `A` of size `(width, height, ncols, nrows)`\ncontaining `width x height` images in a grid of `nrows x ncols`, this returns\na matrix of size `(width * ncols, height * nrows)`, that can be plotted in a heatmap\nto display all images.\n\"\"\"\nfunction imggrid(A::AbstractArray{<:Any,4})\n    reshape(permutedims(A, (1,3,2,4)), size(A,1)*size(A,3), size(A,2)*size(A,4))\nend","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"Now load the MNIST dataset.","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"Float = Float32\ntrain_x, train_y = MLDatasets.MNIST.traindata()\ndigit = 0 # the digit we work with\ntrain_x = Array{Float}(train_x[:, :, train_y .== digit] .> 0.5)\ntrain_y = train_y[train_y .== digit]\nprintln(length(train_y), \" train samples\")","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"Initialize and train a centered RBM","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"rbm = BinaryRBM(Float, (28,28), 200)\nRBMs.initialize!(rbm, train_x)\nrbm = whiten_visible_from_data(whiten(rbm), train_x, WhiteRBMs.Center(); ϵ=1f-3)\nrbm.hidden.θ .= 0\nnothing #hide","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"Pseudolikelihood before training","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"mean(@time RBMs.log_pseudolikelihood(rbm, train_x))","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"Train loop","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"function train!(rbm)\n    batchsize = 256\n    optim = Flux.ADAM()\n    vm = bitrand(28, 28, batchsize) # fantasy chains\n    history = MVHistory()\n    push!(history, :lpl, mean(RBMs.log_pseudolikelihood(rbm, train_x))) # initial log(PL)\n    @time for iter in 1:100 # track pseudolikelihood every 5 epochs\n        WhiteRBMs.pcd!(\n            rbm, train_x; epochs=5, vm, history, batchsize, optim, ϵv=1f-3,\n            transform_v=WhiteRBMs.Center(), transform_h=WhiteRBMs.Center()\n        )\n        push!(history, :lpl, mean(RBMs.log_pseudolikelihood(rbm, train_x)))\n        push!(history, :iter, iter)\n    end\n    return rbm, history, vm\nend","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"Train!","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"rbm, history, vm = train!(rbm)\nnothing #hide","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"Convert to equivalent RBM (without affine transforms)","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"rbm = WhiteRBMs.blacken(rbm)\nnothing #hide","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"Plot log-pseudolikelihood of train data during learning.","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"fig = Makie.Figure(resolution=(600, 300))\nax = Makie.Axis(fig[1,1], xlabel=\"epochs\", ylabel=\"pseudolikelihood\")\nMakie.lines!(ax, get(history, :lpl)...)\nfig","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"Seconds per epoch.","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"fig = Makie.Figure(resolution=(600, 300))\nax = Makie.Axis(fig[1,1], xlabel=\"epoch\", ylabel=\"seconds\")\nMakie.lines!(ax, get(history, :Δt)...)\nfig","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"Log-pseudolikelihood vs. computation time instead of epoch count.","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"fig = Makie.Figure(resolution=(600, 300))\nax = Makie.Axis(fig[1,1], xlabel=\"seconds\", ylabel=\"pseudolikelihood\")\nMakie.lines!(ax, cumsum([0; get(history, :Δt)[2]])[1:5:end], get(history, :lpl)[2])\nfig","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"Now we do the Gibbs sampling to generate RBM digits.","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"nrows, ncols = 10, 15\nnsteps = 5000\nfantasy_F = zeros(nrows*ncols, nsteps)\nfantasy_x = bitrand(28,28,nrows*ncols)\nfantasy_F[:,1] .= RBMs.free_energy(rbm, fantasy_x)\n@time for t in 2:nsteps\n    fantasy_x .= RBMs.sample_v_from_v(rbm, fantasy_x)\n    fantasy_F[:,t] .= RBMs.free_energy(rbm, fantasy_x)\nend\nnothing #hide","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"Check equilibration of sampling","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"fig = Makie.Figure(resolution=(400,300))\nax = Makie.Axis(fig[1,1], xlabel=\"sampling time\", ylabel=\"free energy\")\nfantasy_F_μ = vec(mean(fantasy_F; dims=1))\nfantasy_F_σ = vec(std(fantasy_F; dims=1))\nMakie.band!(ax, 1:nsteps, fantasy_F_μ - fantasy_F_σ/2, fantasy_F_μ + fantasy_F_σ/2)\nMakie.lines!(ax, 1:nsteps, fantasy_F_μ)\nfig","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"Plot the resulting samples.","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"fig = Makie.Figure(resolution=(40ncols, 40nrows))\nax = Makie.Axis(fig[1,1], yreversed=true)\nMakie.image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))\nMakie.hidedecorations!(ax)\nMakie.hidespines!(ax)\nfig","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"","category":"page"},{"location":"literate/MNIST_center/","page":"MNIST centered","title":"MNIST centered","text":"This page was generated using Literate.jl.","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"EditURL = \"https://github.com/cossio/WhitenedRBMs.jl/blob/master/docs/src/literate/MNIST_identity.jl\"","category":"page"},{"location":"literate/MNIST_identity/#MNIST","page":"MNIST identity","title":"MNIST","text":"","category":"section"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"We begin by importing the required packages. We load MNIST via the MLDatasets.jl package.","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"Some references","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"https://www.jmlr.org/beta/papers/v17/14-237.html,","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"http://www.cs.toronto.edu/~tang/papers/RbmZM.pdf,","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"https://doi.org/10.1007/978-3-642-35289-8_3","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"import CairoMakie\nimport Makie\nimport MLDatasets\nimport Flux\nimport RestrictedBoltzmannMachines as RBMs\nimport WhitenedRBMs as WhiteRBMs\nusing Statistics: mean, var, std\nusing Random: bitrand\nusing ValueHistories: MVHistory\nusing RestrictedBoltzmannMachines: BinaryRBM\nusing WhitenedRBMs: whiten, blacken, whiten_visible_from_data\nnothing #hide","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"Useful function to plot MNIST digits.","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"\"\"\"\n    imggrid(A)\n\nGiven a four dimensional tensor `A` of size `(width, height, ncols, nrows)`\ncontaining `width x height` images in a grid of `nrows x ncols`, this returns\na matrix of size `(width * ncols, height * nrows)`, that can be plotted in a heatmap\nto display all images.\n\"\"\"\nfunction imggrid(A::AbstractArray{<:Any,4})\n    reshape(permutedims(A, (1,3,2,4)), size(A,1)*size(A,3), size(A,2)*size(A,4))\nend","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"Now load the MNIST dataset.","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"Float = Float32\ntrain_x, train_y = MLDatasets.MNIST.traindata()\ndigit = 0 # the digit we work with\ntrain_x = Array{Float}(train_x[:, :, train_y .== digit] .> 0.5)\ntrain_y = train_y[train_y .== digit]\nprintln(length(train_y), \" train samples\")","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"Initialize and train a normal RBM","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"rbm = BinaryRBM(Float, (28,28), 200)\nRBMs.initialize!(rbm, train_x)\nrbm = whiten_visible_from_data(whiten(rbm), train_x, WhiteRBMs.Identity(); ϵ=1f-3)\nrbm.hidden.θ .= 0\nnothing #hide","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"Train init","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"batchsize = 256\noptim = Flux.ADAM()\nvm = bitrand(28, 28, batchsize) # fantasy chains\nhistory = MVHistory()\npush!(history, :lpl, mean(RBMs.log_pseudolikelihood(rbm, train_x)))\nnothing #hide","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"Pseudolikelihood before training","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"mean(@time RBMs.log_pseudolikelihood(rbm, train_x))","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"Train","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"@time for epoch in 1:100 # track pseudolikelihood every 5 epochs\n    WhiteRBMs.pcd!(\n        rbm, train_x; epochs=5, vm, history, batchsize, optim, ϵv=1f-3,\n        transform_v=WhiteRBMs.Identity(), transform_h=WhiteRBMs.Identity()\n    )\n    push!(history, :lpl, mean(RBMs.log_pseudolikelihood(rbm, train_x)))\nend\nnothing #hide","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"Convert to equivalent RBM (without affine transforms)","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"rbm = WhiteRBMs.blacken(rbm)\nnothing #hide","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"Plot log-pseudolikelihood of train data during learning.","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"fig = Makie.Figure(resolution=(600, 300))\nax = Makie.Axis(fig[1,1], xlabel=\"epochs\", ylabel=\"pseudolikelihood\")\nMakie.lines!(ax, get(history, :lpl)...)\nfig","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"Seconds per epoch.","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"fig = Makie.Figure(resolution=(600, 300))\nax = Makie.Axis(fig[1,1], xlabel=\"epoch\", ylabel=\"seconds\")\nMakie.lines!(ax, get(history, :Δt)...)\nfig","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"Log-pseudolikelihood vs. computation time instead of epoch count.","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"fig = Makie.Figure(resolution=(600, 300))\nax = Makie.Axis(fig[1,1], xlabel=\"seconds\", ylabel=\"pseudolikelihood\")\nMakie.lines!(ax, cumsum([0; get(history, :Δt)[2]])[1:5:end], get(history, :lpl)[2])\nfig","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"Now we do the Gibbs sampling to generate RBM digits.","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"nrows, ncols = 10, 15\nnsteps = 5000\nfantasy_F = zeros(nrows*ncols, nsteps)\nfantasy_x = bitrand(28,28,nrows*ncols)\nfantasy_F[:,1] .= RBMs.free_energy(rbm, fantasy_x)\n@time for t in 2:nsteps\n    fantasy_x .= RBMs.sample_v_from_v(rbm, fantasy_x)\n    fantasy_F[:,t] .= RBMs.free_energy(rbm, fantasy_x)\nend\nnothing #hide","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"Check equilibration of sampling","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"fig = Makie.Figure(resolution=(400,300))\nax = Makie.Axis(fig[1,1], xlabel=\"sampling time\", ylabel=\"free energy\")\nfantasy_F_μ = vec(mean(fantasy_F; dims=1))\nfantasy_F_σ = vec(std(fantasy_F; dims=1))\nMakie.band!(ax, 1:nsteps, fantasy_F_μ - fantasy_F_σ/2, fantasy_F_μ + fantasy_F_σ/2)\nMakie.lines!(ax, 1:nsteps, fantasy_F_μ)\nfig","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"Plot the resulting samples.","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"fig = Makie.Figure(resolution=(40ncols, 40nrows))\nax = Makie.Axis(fig[1,1], yreversed=true)\nMakie.image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))\nMakie.hidedecorations!(ax)\nMakie.hidespines!(ax)\nfig","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"","category":"page"},{"location":"literate/MNIST_identity/","page":"MNIST identity","title":"MNIST identity","text":"This page was generated using Literate.jl.","category":"page"},{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [WhitenedRBMs]","category":"page"},{"location":"reference/#RestrictedBoltzmannMachines.RBM-Tuple{WhitenedRBMs.WhiteRBM}","page":"Reference","title":"RestrictedBoltzmannMachines.RBM","text":"RBM(white_rbm::WhiteRBM)\n\nReturns an (unwhitened) RBM which neglects the affine transforms of white_rbm. The resulting model is not equivalent to the original white_rbm. To construct an equivalent model, use the function blacken(white_rbm) instead (see blacken).\n\n\n\n\n\n","category":"method"},{"location":"reference/#WhitenedRBMs.WhiteRBM-Tuple{RestrictedBoltzmannMachines.RBM}","page":"Reference","title":"WhitenedRBMs.WhiteRBM","text":"WhiteRBM(rbm)\n\nCreates a WhiteRBM with identity transforms.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.one-Tuple{WhitenedRBMs.Affine}","page":"Reference","title":"Base.one","text":"one(t::Affine)\n\nReturns the identity affine transformation, which maps all points to themselves.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.zero-Tuple{WhitenedRBMs.Affine}","page":"Reference","title":"Base.zero","text":"zero(t::Affine)\n\nReturns the zero affine transformation, which maps all points to zero.\n\n\n\n\n\n","category":"method"},{"location":"reference/#WhitenedRBMs.BinaryWhiteRBM-Tuple{AbstractArray, AbstractArray, AbstractArray, WhitenedRBMs.Affine, WhitenedRBMs.Affine}","page":"Reference","title":"WhitenedRBMs.BinaryWhiteRBM","text":"BinaryWhiteRBM(a, b, w, affine_v, affine_h)\nBinaryWhiteRBM(a, b, w)\n\nConstruct a whitened RBM with binary visible and hidden units.\n\n\n\n\n\n","category":"method"},{"location":"reference/#WhitenedRBMs.blacken-Tuple{WhitenedRBMs.WhiteRBM}","page":"Reference","title":"WhitenedRBMs.blacken","text":"blacken(white_rbm::WhiteRBM)\n\nConstructs a plain RBM equivalent to the given white_rbm. The energies assigned by the two models differ by a constant amount,\n\nE(vh) - tildeE(vh) = mathbfa^top mathbbA^top tildemathbbW mathbbBmathbfb\n\nwhere tildeE(vh) is the energy assigned by white_rbm and E(vh) is the energy assigned by the RBM constructed by this method.\n\nThis is the inverse operation of whiten, always returning an ordinary RBM.\n\nTo construct an RBM that simply neglects the transformations, call RBM(white_rbm) instead.\n\n\n\n\n\n","category":"method"},{"location":"reference/#WhitenedRBMs.energy_shift-Tuple{WhitenedRBMs.WhiteRBM, WhitenedRBMs.Affine, WhitenedRBMs.Affine}","page":"Reference","title":"WhitenedRBMs.energy_shift","text":"energy_shift(rbm, affine_v, affine_h)\n\nComputes the constant energy shift if the affine transformations were updated as given.\n\n\n\n\n\n","category":"method"},{"location":"reference/#WhitenedRBMs.hidden_affine_from_inputs","page":"Reference","title":"WhitenedRBMs.hidden_affine_from_inputs","text":"hidden_affine_from_inputs(white_rbm, data; wts = nothing, ϵ = 0)\n\nReturns the affine transformation that standardizes the hidden inputs. Note that only the variances are scaled to unity. The correlation matrix is not diagonalized.\n\n\n\n\n\n","category":"function"},{"location":"reference/#WhitenedRBMs.safe_whiten-Union{Tuple{Ah}, Tuple{Av}, Tuple{WhitenedRBMs.WhiteRBM{V, H, W, Av, Ah} where {V, H, W}, Av, Ah}} where {Av, Ah}","page":"Reference","title":"WhitenedRBMs.safe_whiten","text":"safe_whiten(white_rbm, affine_v, affine_h)\n\nLike whiten(white_rbm, affine_v, affine_h), but ensures that the returned WhiteRBM is of the same type as white_rbm (in particular, preserving the types of the affine transformations).\n\n\n\n\n\n","category":"method"},{"location":"reference/#WhitenedRBMs.shift_fields","page":"Reference","title":"WhitenedRBMs.shift_fields","text":"shift_fields(layer, offset)\n\nAdds offset to the layer fields.\n\n\n\n\n\n","category":"function"},{"location":"reference/#WhitenedRBMs.shift_fields!","page":"Reference","title":"WhitenedRBMs.shift_fields!","text":"shift_fields!(layer, offset)\n\nIn-place version of shift_fields(layer, offset).\n\n\n\n\n\n","category":"function"},{"location":"reference/#WhitenedRBMs.visible_affine_from_data","page":"Reference","title":"WhitenedRBMs.visible_affine_from_data","text":"visible_affine_from_data(white_rbm, data; wts = nothing, ϵ = 0)\n\nReturns the affine transformation that whitens the visible data.\n\n\n\n\n\n","category":"function"},{"location":"reference/#WhitenedRBMs.whiten-Tuple{RestrictedBoltzmannMachines.RBM, WhitenedRBMs.Affine, WhitenedRBMs.Affine}","page":"Reference","title":"WhitenedRBMs.whiten","text":"whiten(rbm::RBM, affine_v, affine_h)\n\nConstructs a WhiteRBM equivalent to the given rbm. The energies assigned by the two models differ by a constant amount,\n\nE(vh) - tildeE(vh) = mathbfa^top mathbbA^top tildemathbbW mathbbBmathbfb\n\nwhere E(vh) is the energy assigned by the original rbm, and tildeE(vh) is the energy assigned by the returned WhiteRBM.\n\nThis is the inverse operation of blacken, always returning a WhiteRBM.\n\nTo construct a WhiteRBM that simply includes these transforms, call WhiteRBM(rbm, affine_v, affine_h) instead.\n\n\n\n\n\n","category":"method"},{"location":"reference/#WhitenedRBMs.whitening_transform-Tuple{AbstractVector, AbstractMatrix}","page":"Reference","title":"WhitenedRBMs.whitening_transform","text":"whitening_transform(μ, C)\n\nReturns the Affine transform that whitens data with mean μ and covariance C.\n\n\n\n\n\n","category":"method"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"EditURL = \"https://github.com/cossio/WhitenedRBMs.jl/blob/master/docs/src/literate/MNIST_white.jl\"","category":"page"},{"location":"literate/MNIST_white/#MNIST","page":"MNIST whitened","title":"MNIST","text":"","category":"section"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"We begin by importing the required packages. We load MNIST via the MLDatasets.jl package.","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"Some references","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"https://www.jmlr.org/beta/papers/v17/14-237.html,","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"http://www.cs.toronto.edu/~tang/papers/RbmZM.pdf,","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"https://doi.org/10.1007/978-3-642-35289-8_3","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"import CairoMakie\nimport Makie\nimport MLDatasets\nimport Flux\nimport RestrictedBoltzmannMachines as RBMs\nimport WhitenedRBMs as WhiteRBMs\nusing Statistics: mean, var, std\nusing Random: bitrand\nusing ValueHistories: MVHistory\nusing RestrictedBoltzmannMachines: BinaryRBM\nusing WhitenedRBMs: whiten, blacken, whiten_visible_from_data\nnothing #hide","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"Useful function to plot MNIST digits.","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"\"\"\"\n    imggrid(A)\n\nGiven a four dimensional tensor `A` of size `(width, height, ncols, nrows)`\ncontaining `width x height` images in a grid of `nrows x ncols`, this returns\na matrix of size `(width * ncols, height * nrows)`, that can be plotted in a heatmap\nto display all images.\n\"\"\"\nfunction imggrid(A::AbstractArray{<:Any,4})\n    reshape(permutedims(A, (1,3,2,4)), size(A,1)*size(A,3), size(A,2)*size(A,4))\nend","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"Now load the MNIST dataset.","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"Float = Float32\ntrain_x, train_y = MLDatasets.MNIST.traindata()\ndigit = 0 # the digit we work with\ntrain_x = Array{Float}(train_x[:, :, train_y .== digit] .> 0.5)\ntrain_y = train_y[train_y .== digit]\nprintln(length(train_y), \" train samples\")","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"Initialize and train a whitened RBM","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"rbm = BinaryRBM(Float, (28,28), 200)\nRBMs.initialize!(rbm, train_x)\nrbm = whiten_visible_from_data(whiten(rbm), train_x, WhiteRBMs.Whiten(); ϵ=1f-3)\nrbm.hidden.θ .= 0\nnothing #hide","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"Train init","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"batchsize = 256\noptim = Flux.ADAM()\nvm = bitrand(28, 28, batchsize) # fantasy chains\nhistory = MVHistory()\npush!(history, :lpl, mean(RBMs.log_pseudolikelihood(rbm, train_x)))\nnothing #hide","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"Pseudolikelihood before training","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"mean(@time RBMs.log_pseudolikelihood(rbm, train_x))","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"Train","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"@time for epoch in 1:100 # track pseudolikelihood every 5 epochs\n    WhiteRBMs.pcd!(\n        rbm, train_x; epochs=5, vm, history, batchsize, optim, ϵv=1f-3,\n        transform_v = WhiteRBMs.Whiten(), transform_h = WhiteRBMs.Stdize()\n    )\n    push!(history, :lpl, mean(RBMs.log_pseudolikelihood(rbm, train_x)))\nend\nnothing #hide","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"Convert to equivalent RBM (without affine transforms)","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"rbm = WhiteRBMs.blacken(rbm)\nnothing #hide","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"Plot log-pseudolikelihood of train data during learning.","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"fig = Makie.Figure(resolution=(600, 300))\nax = Makie.Axis(fig[1,1], xlabel=\"epochs\", ylabel=\"pseudolikelihood\")\nMakie.lines!(ax, get(history, :lpl)...)\nfig","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"Seconds per epoch.","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"fig = Makie.Figure(resolution=(600, 300))\nax = Makie.Axis(fig[1,1], xlabel=\"epoch\", ylabel=\"seconds\")\nMakie.lines!(ax, get(history, :Δt)...)\nfig","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"Log-pseudolikelihood vs. computation time instead of epoch count.","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"fig = Makie.Figure(resolution=(600, 300))\nax = Makie.Axis(fig[1,1], xlabel=\"seconds\", ylabel=\"pseudolikelihood\")\nMakie.lines!(ax, cumsum([0; get(history, :Δt)[2]])[1:5:end], get(history, :lpl)[2])\nfig","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"Now we do the Gibbs sampling to generate RBM digits.","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"nrows, ncols = 10, 15\nnsteps = 5000\nfantasy_F = zeros(nrows*ncols, nsteps)\nfantasy_x = bitrand(28,28,nrows*ncols)\nfantasy_F[:,1] .= RBMs.free_energy(rbm, fantasy_x)\n@time for t in 2:nsteps\n    fantasy_x .= RBMs.sample_v_from_v(rbm, fantasy_x)\n    fantasy_F[:,t] .= RBMs.free_energy(rbm, fantasy_x)\nend\nnothing #hide","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"Check equilibration of sampling","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"fig = Makie.Figure(resolution=(400,300))\nax = Makie.Axis(fig[1,1], xlabel=\"sampling time\", ylabel=\"free energy\")\nfantasy_F_μ = vec(mean(fantasy_F; dims=1))\nfantasy_F_σ = vec(std(fantasy_F; dims=1))\nMakie.band!(ax, 1:nsteps, fantasy_F_μ - fantasy_F_σ/2, fantasy_F_μ + fantasy_F_σ/2)\nMakie.lines!(ax, 1:nsteps, fantasy_F_μ)\nfig","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"Plot the resulting samples.","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"fig = Makie.Figure(resolution=(40ncols, 40nrows))\nax = Makie.Axis(fig[1,1], yreversed=true)\nMakie.image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))\nMakie.hidedecorations!(ax)\nMakie.hidespines!(ax)\nfig","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"","category":"page"},{"location":"literate/MNIST_white/","page":"MNIST whitened","title":"MNIST whitened","text":"This page was generated using Literate.jl.","category":"page"},{"location":"#WhitenedRBMs.jl-Documentation","page":"Home","title":"WhitenedRBMs.jl Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A Julia package to train and simulate whitened Restricted Boltzmann Machines.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package does not export any symbols.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Most of the functions have a helpful docstring. See Reference section.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package is not registered. Install with:","category":"page"},{"location":"","page":"Home","title":"Home","text":"import Pkg\nPkg.add(url=\"https://github.com/cossio/WhitenedRBMs.jl\")","category":"page"},{"location":"#Related","page":"Home","title":"Related","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package is based on the RestrictedBoltzmannMachines Julia package, which defines the RBM and layer types. We refer to RestrictedBoltzmannMachines by the shorter name RBMs, as if it were imported by the line","category":"page"},{"location":"","page":"Home","title":"Home","text":"import RestrictedBoltzmannMachines as RBMs","category":"page"}]
}
