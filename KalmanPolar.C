void KalmanPolar()
{
	int n = 1000;
	double x[n];
	double y[n];
	double ex[n];
	double ey[n];

	//TGraphErrors *gr = new TGraphErrors();

	fstream file;
	file.open("2dKalmanModelTrust.txt", ios::in);

	for(int i=0;i<1000;i++){
		file >> x[i] >> y[i] >> ex[i] >> ey[i];
		//gr->SetPoint(gr->GetN(),x[i],y[i],ex[i],ey[i]);
		std::cout << x[i] << "\t" << y[i] << "\t" << "\n";
	}
	file.close();

	TGraphErrors *gr = new TGraphErrors(n,x,y,ex,ey);

	gr->SetTitle("Kalman Filter");
	gr->SetMarkerStyle(kFullCircle);
	gr->SetMarkerSize(.5);
	gr->SetLineColor(2);
	gr->SetLineWidth(0.1);

	TCanvas *c1 = new TCanvas();
	c1->Draw();	
	c1->SetCanvasSize(1000,1000);
	c1->SetWindowSize(500,500);
	
	TH1F* histo = new TH1F("histo","Kalman Filter Trusting Model",1,-10.5,10.5);
	histo->SetLineWidth(0);
	histo->Draw();
	histo->GetYaxis()->SetRangeUser(-10.5,10.5);	

	for (int i=1;i<11;i++) {
		TEllipse* e = new TEllipse(0.0,0.0,11-1*i);
		e->Draw("SAME");	
	}

	gr->Draw("P same");
	
	gStyle->SetOptStat(0);
	//gr->Draw("APE");
	c1->Update();
}	
