function action(input, output, filename) {
        open(input + filename);
		run("Histogram");
		//getting histogram results
		getHistogram(value, count, 256);
		//the first parameter for image - seperate diagnoses label
		if(endsWith(filename, "_nv.jpg")){
		File.append("3", "C:/Users/Brigita/Desktop/Bakalaurinis/histogramu_rezultatai.csv");}
		else if(endsWith(filename, "_mel.jpg")){
		File.append("4", "C:/Users/Brigita/Desktop/Bakalaurinis/histogramu_rezultatai.csv");}
		else if(endsWith(filename, "_bkl.jpg")){
		File.append("1", "C:/Users/Brigita/Desktop/Bakalaurinis/histogramu_rezultatai.csv");}
		else if(endsWith(filename, "_akiec.jpg")){
		File.append("0", "C:/Users/Brigita/Desktop/Bakalaurinis/histogramu_rezultatai.csv");}
		else if(endsWith(filename, "_df.jpg")){
		File.append("2", "C:/Users/Brigita/Desktop/Bakalaurinis/histogramu_rezultatai.csv");}
		else if(endsWith(filename, "_bcc.jpg")){
		File.append("5", "C:/Users/Brigita/Desktop/Bakalaurinis/histogramu_rezultatai.csv");}
		else if(endsWith(filename, "_vasc.jpg")){
		File.append("6", "C:/Users/Brigita/Desktop/Bakalaurinis/histogramu_rezultatai.csv");}
		//after label - histogram results parameters of the image
		for (i=0; i<256; i++){
			setResult("Count", i, count[i]);
			File.append(count[i], "C:/Users/Brigita/Desktop/Bakalaurinis/histogramu_rezultatai.csv") 
		}		
		updateResults();
			 
		close();
		close();
}
input = "C:/Users/Brigita/Desktop/Bakalaurinis/HAM10000_images/";

//list of all images
list = getFileList(input);
//calling function
for (i = 0; i < list.length; i++)
        action(input, output, list[i]);