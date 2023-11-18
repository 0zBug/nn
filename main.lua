
local nn = {}

local BaseMap = {}

local function Lookup(Table)
	local Lookup = {}

	for Index, Value in next, Table do
		Lookup[Value] = Index
	end

	return Lookup
end

function BaseMap.new(Map)
	local N = #Map + 1
	local LookupMap = Lookup(Map)

	local Max, Total, Limit = 0, 0, nil
	
	repeat 
		Max = Max + 1
		Total = Total + N ^ Max
		Limit = Total >= 0x10FFFF
	until Limit

	Max = Max - 1

	local Base = {}

	function Base.Encode(String)
		return string.gsub(String, "." .. string.rep(".?", Max - 1), function(String)
			local Number = 0

			for Index = 1, Max do
				local Cut = string.sub(String, Index, Index)

				if Cut ~= "" then
					Number = Number + LookupMap[Cut] * N ^ (Index - 1)
				end
			end

			return utf8.char(Number)
		end)
	end

	function Base.Decode(String)
		return string.gsub(String, "([%z\1-\127\194-\244][\128-\191]*)", function(String)
			local Number

			for i = 1, #String do
				local c = string.byte(String, i)

				Number = (i == 1) and bit32.band(c, (2 ^ (8 - (c < 0x80 and 1 or c < 0xE0 and 2 or c < 0xF0 and 3 or c < 0xF8 and 4 or c < 0xFC and 5 or c < 0xFE and 6)) - 1)) or (bit32.lshift(Number, 6) + bit32.band(c, 0x3F))
			end

			local Decoded = {}

			for Index = 1, Max do
				local n = math.floor((Number / (N ^ (Index - 1))) % N)

				table.insert(Decoded, Map[n])

				Number = Number - n
			end

			return table.concat(Decoded)
		end)
	end

	return Base
end

local Base = BaseMap.new({"-", ".", " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})

function nn.new(layers, rate, threshold)
	local network = {rate = rate, threshold = threshold}

	for i = 1, #layers do
		local cells = {}

		for c = 1, layers[i] or 1 do
			local cell = {delta = 0, weights = {}, signal = 0}
			
			for w = 1, layers[i - 1] or 1 do
				cell.weights[w] = math.random()
			end 
			
			cell.activate = function(inputs, bias, threshold)
				local sum = bias
				local weights = cell.weights
				
				for w, weight in weights do
					sum = sum + weight * inputs[w]
				end
				
				cell.signal = 1 / (1 + math.exp(-sum) / threshold)
			end
			
			cells[c] = cell
		end

		network[i] = {cells = cells, bias = math.random()}
	end

	function network.predict(input)
		local threshold = network.threshold
		
		for i = 1, #input do
			network[1].cells[i].signal = input[i]
		end
		
		for l = 2, #network do
			local input = {}
			
			local cells = network[l].cells
			local previous = network[l - 1].cells
			
			for c = 1, #previous do
				input[c] = previous[c].signal
			end
			
			for _, cell in cells do
				cell.activate(input, network[l].bias, threshold)
			end
		end
		
		local prediction = {}
		
		for i = 1, #network[#network].cells do
			table.insert(prediction, network[#network].cells[i].signal)
		end
		
		return prediction
	end
	
	function network.train(input, output)
		network.predict(input)

		for l = #network, 2, -1 do
			local cells = network[l].cells
			
			for c = 1, #cells do
				local signal = cells[c].signal
				
				if l ~= #network then
					local weight = 0
					local layer = network[l + 1].cells
					
					for k = 1, #network[l + 1].cells do
						weight = weight + layer[k].weights[c] * layer[k].delta
					end
					
					cells[c].delta = signal * (1 - signal) * weight
				else
					cells[c].delta = (output[c] - signal) * signal * (1 - signal)
				end
			end
		end

		for l = 2, #network do
			local cells = network[l].cells
			
			network[l].bias = cells[#cells].delta * network.rate
			
			for _, cell in cells do
				for w = 1, #cell.weights do
					local weights = cell.weights
					
					weights[w] = weights[w] + cell.delta * network.rate * network[l - 1].cells[w].signal
				end
			end
		end
	end
	
	function network.export()
		local package = {#layers, rate, threshold}

		for l = 1, #layers do
			table.insert(package, layers[l])
			table.insert(package, network[l].bias)
		end
		
		for l = 1, #layers do
			local weights = {}
			
			for c = 1, layers[l] do
				local cell = network[l].cells[c]

				for w = 1, #cell.weights do
					local weight = cell.weights[w]

					table.insert(weights, weight)
				end
			end
			
			table.insert(package, #network[l].cells[1].weights)
			table.insert(package, l)
			table.insert(package, #network[l].cells)
			
			for i = 1, #weights do
				table.insert(package, weights[i])
			end
		end
		
		return Base.Encode(table.concat(package, " "))
	end
	
	return network
end

function nn.import(data)
	local data = string.split(Base.Decode(data), " ")
	
	local function read()
		local item = tonumber(data[1])
		table.remove(data, 1)
		
		return item
	end
	local nlayers = read()
	local rate = read()
	local threshold = read()
	
	local layers = {}
	local biases = {}

	for l = 1, nlayers do
		table.insert(layers, read())
		table.insert(biases, read())
	end

	local network = nn.new(layers, rate, threshold)
	network.threshold = threshold
	network.rate = rate
	
	for i = 1, #biases do
		network[i].bias = biases[i]
	end
	
	while data[1] do
		local weights, layer, cells = read(), read(), read()

		for c = 1, cells do
			for w = 1, weights do
				network[layer].cells[c].weights[w] = read()
			end
		end
	end

	return network
end

return nn
