void ConcentricKalman()
{
	int n = 100; //Number of positions, 10 for each particle
	int nd = 10; //Number of data points per particle/track to be fitted
	int fc = 2; //Number of coefficients from the fit to plot a function

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
	fstream coef_file;
	file.open("2dKalmanDataSameError.txt", ios::in);
	coef_file.open("coef-fit10.txt", ios::in);
	
	
	for(int i = 0 ; i < (n/nd) ; i++){ //Loop taking data to vectors, each iteration reconstructs one particle track
		double x[nd];	  //Vectors which will store positions and errors
		double y[nd];
		double ex[nd];
		double ey[nd];
		double coef[fc]; //Vector with the polynomial coefficients of the fitted track

		for (int  j = 0 ; j < nd ; j++){
			file >> x[j] >> y[j] >> ex[j] >> ey[j];
			//std::cout << x[j] << "\t" << y[j] << "\t" << "\n";
		}
				
		TGraphErrors *gr = new TGraphErrors(nd,x,y,ex,ey); //Graph initialization to plot the track

		//TGraphErrors *gr = new TGraphErrors("2dKalmanDataSameError.txt","%lg    %lg    %lg    %lg");

		gr->SetMarkerStyle(kFullCircle);
		gr->SetMarkerSize(.5);
		gr->SetLineColor(2);
		gr->SetLineWidth(0.1);

		gr->Draw("P same"); //Drawing points over the circles

		TF1 *fa1 = new TF1("fa1","[0]+[1]*x",0,10);

		coef_file >> coef[0] >> coef[1];

		std::cout << coef[1] << "\t" << coef[0] << std::endl;

		fa1->SetParameter(0,coef[1]);
		fa1->SetParameter(1,coef[0]);

		fa1->Draw("L same");
	}

	file.close();
	coef_file.close();

	gStyle->SetOptStat(0); //Don't show histogram statistics box
	c1->Update();
}	
