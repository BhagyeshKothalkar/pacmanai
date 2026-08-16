what is my input?
my input is- take the current image (and its underlying state) and the next state, regenerate the next image.
this is basically matting, but the thing is that that is harder than what i want;
that is, because the new state is known, all i have to do is the next image.
what is the easiest problem here?
the easiest problem here is when no pixel actually leaves the screen or enters it, and all move smoothly.
maybe if there is just some particle swarm and where you move actually works out where you have to place the next object.

so if you dont rotate things the in pacman or dont change colors (so the sprites remain the same throughout), the task is just to move them, and make food disappear.
ok, so just moving points.
given previous points, the previous pixels, the next points, sprites, generate the next image.
now, the mechanical way is to put those sprites at those ponts, 
flux would just take the points, and the sprites and generate the image.
what i was thinking towards, is ki move the structures in the current image.


so the plan  is: give them the individual movements and not the locations. 
and how do i give the movements? 
the pacman moves, the ghosts move
i just have the  move  they make. 
so what? 
i have the moes they make. 
so rather do: the pacman moves, the ghosts moev, the palette gets eaten. 