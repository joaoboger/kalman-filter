void ConcentricKalman()
{
	int n = 1000; //Number of positions, 10 for each particle
	int nd = 10; //Number of data points per particle/track to be fitted

	TCanvas *c1 = new TCanvas(); //Canvas initialization
	c1->Draw();	
	c1->SetCanvasSize(1000,1000);
	c1->SetWindowSize(500,500);
	
	TH1F* histo = new TH1F("histo","Kalman Filter",1,-10.5,10.5); //Creating histogram in order to control axis' range

	histo->SetLineWidth(0);
	histo->Draw();
	histo->GetYaxis()->SetRangeUser(-10.5,10.5);	

	for (int i=1;i<11;i++) { //Loop drawing concentric circles
		TEllipse* e = new TEllipse(0.0,0.0,11-1*i);
		e->Draw("SAME");	
	}

	fstream file; //Initialization of the data
	file.open("2dKalmanSameError.txt", ios::in);
	
	
	for(int i = 0 ; i < (n/nd) ; i++){ //Loop taking data to vectors, each iteration reconstructs one particle track
		double x[nd];	  //Vectors which will store positions and errors
		double y[nd];
		double ex[nd];
		double ey[nd];
		for (int  j = 0 ; j < nd ; j++){
			file >> x[j] >> y[j] >> ex[j] >> ey[j];
			std::cout << x[j] << "\t" << y[j] << "\t" << "\n";
		}
				
		TGraphErrors *gr = new TGraphErrors(nd,x,y,ex,ey); //Graph initialization to plot the track

		gr->SetMarkerStyle(kFullCircle);
		gr->SetMarkerSize(.5);
		gr->SetLineColor(2);
		gr->SetLineWidth(0.1);

		gr->Draw("P same"); //Drawing points over the circles

		gr->Fit("pol1","+","L same",0,10.5); //Fits our tracks in a line (polynomial of first order). The fit must be set according to the problem being considered
	}

	file.close();

	
	gStyle->SetOptStat(0); //Don't show histogram statistics box
	c1->Update();
}	
