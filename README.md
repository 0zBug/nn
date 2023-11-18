```lua
local net = nn.new({2, 64, 1}, 1, 0.1) -- layers, learning rate, threshold

print(net)

for i = 1, 10000 do
	net.train({0, 0}, {0})
	net.train({1, 0}, {1})
	net.train({0, 1}, {1})
	net.train({1, 1}, {0})
end

print("{0, 0}", "{" .. table.concat(net.predict({0, 0}), ", ") .. "}")
print("{0, 1}", "{" .. table.concat(net.predict({0, 1}), ", ") .. "}")
print("{1, 0}", "{" .. table.concat(net.predict({1, 0}), ", ") .. "}")
print("{1, 1}", "{" .. table.concat(net.predict({1, 1}), ", ") .. "}")

local net = nn.import(net.export())

print("{0, 0}", "{" .. table.concat(net.predict({0, 0}), ", ") .. "}")
print("{0, 1}", "{" .. table.concat(net.predict({0, 1}), ", ") .. "}")
print("{1, 0}", "{" .. table.concat(net.predict({1, 0}), ", ") .. "}")
print("{1, 1}", "{" .. table.concat(net.predict({1, 1}), ", ") .. "}")

print(#net.export())
```

(trash)
