
struct Padding
    padding::Int
    top::Int
    bottom::Int
    left::Int
    right::Int

    function Padding(padding::Int)
        return new(
            padding,
            0, 
            0,
            0,
            0
        )
    end
end
