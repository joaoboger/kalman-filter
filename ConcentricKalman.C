void KalmanPolar()
{
	int n = 1000; //Number of positions, 10 for each particle
	double x[n]; //Vectors which will store positions and errors 
	double y[n];
	double ex[n];
	double ey[n];

	fstream file; //Initialization of the data
	file.open("2dKalmanModelTrust.txt", ios::in);
	
	
	for(int i=0;i<1000;i++){ //Loop taking data to vectors
		file >> x[i] >> y[i] >> ex[i] >> ey[i];
		std::cout << x[i] << "\t" << y[i] << "\t" << "\n";
	}

	file.close();

	TGraphErrors *gr = new TGraphErrors(n,x,y,ex,ey); //Graph in which data will be plotted

	gr->SetTitle("Kalman Filter");
	gr->SetMarkerStyle(kFullCircle);
	gr->SetMarkerSize(.5);
	gr->SetLineColor(2);
	gr->SetLineWidth(0.1);

	TCanvas *c1 = new TCanvas(); //Canvas initialization
	c1->Draw();	
	c1->SetCanvasSize(1000,1000);
	c1->SetWindowSize(500,500);
	
	TH1F* histo = new TH1F("histo","Kalman Filter Trusting Model",1,-10.5,10.5); //Creating histogram in order to control axis' range
	histo->SetLineWidth(0);
	histo->Draw();
	histo->GetYaxis()->SetRangeUser(-10.5,10.5);	

	for (int i=1;i<11;i++) { //Loop drawing concentric circles
		TEllipse* e = new TEllipse(0.0,0.0,11-1*i);
		e->Draw("SAME");	
	}

	gr->Draw("P same"); //Drawing points over the circles
	
	gStyle->SetOptStat(0); //Don't show histogram statistics box
	c1->Update();
}	
