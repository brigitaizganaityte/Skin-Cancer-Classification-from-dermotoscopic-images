function action(input, filename) {
        open(input + filename);
		
		updateResults();
		//saveAs("Results", output + "Histogram of " + filename + ".csv");
		 setOption("BlackBackground", false);
		run("Make Binary");
		makeRectangle(0, 0, 600, 450);
		run("Gaussian Blur...", "sigma=3");
		run("Save XY Coordinates...", "save=C:/Users/Brigita/Desktop/Bakalaurinis/findEdgeData/"+filename+".csv");
		close();
		}
input = "C:/Users/Brigita/Desktop/Bakalaurinis/HAM10000_images/";


list = getFileList(input);
for (i = 0; i < list.length; i++)
        action(input, list[i]);
		
		
