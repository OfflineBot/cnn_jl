
struct Padding
    padding::Int
    top::Int
    bottom::Int
    left::Int
    right::Int

    function Padding(padding::Int)
        new(
            padding,
            0, 
            0,
            0,
            0
        )
    end
end
