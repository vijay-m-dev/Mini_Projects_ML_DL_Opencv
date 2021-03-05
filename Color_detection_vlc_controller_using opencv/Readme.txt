VLC Controller using Hot Keys using Opencv
	- Used to control the vlc player using any coloured(Red,Sky Blue) Objects positions infront of webcam using opencv and Pyautogui

Description:
	- The window is divided into Upper left,Lower Left,Lower Left,Lower Right.
	  If the Sky blue color is showed in front of webcam, the video gets played or paused.
	  If the Red object is showed in front of webcam, the video gets volume up,down or the video gets rewind,forward.
	  Red Object positions infront of webcam:
		- Upper left:Video rewind
		- Lower left:Video forward
		- Upper right:Volume up
		- Lower right:Volume down
	- The Hotkeys of arrow buttons are called using Pyautogui.

Note:
	- After running the code, the video should be played manually by us using vlc player.
	- To stop the code, select the table which shows the webcam and press 'q'.