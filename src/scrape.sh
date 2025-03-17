firefox "$1" && sleep 1 \
&& xdotool mousemove 158 749 && xdotool click 3 && xdotool mousemove 262 510 \
&& xdotool click 1 && sleep .3 \
&& xdotool type "~/stock/src/page1.html" && xdotool key 0xff09 && xdotool key 0xff54 \
&& xdotool key --repeat=2 0xff09 && xdotool key Return && xdotool key Tab && xdotool key Return\
&& sleep .7 && xdotool key Return \
&& sleep 0.2 && xdotool key --clearmodifiers Alt+F4