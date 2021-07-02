from math import sqrt, pow, sin, cos, pi, ceil, atan
import ROOT as rt
import ctypes

### Calculate the track of a charged particle in a magnetic field.
### The particle is assumed to go through the origin of the coordinate system,
### and cross a number of concentric detector layers at constant radius.
### It leaves a hit on every layer, with an optional error on the position of the hit.
### (since we assume it goes through the origin, the curvature radius is fixed.)
### We use polar coordinates (rho, phi) throughout.
###  * sign: sign of the particle charge (assumed to be either +1 or -1)
###  * layerDistance: distance between consecutive layers
###  * numLayers: number of concentric layers
###  * errorPhi: error in the phi coordinate of the hit


def makeTrack(centerX, centerY, sign, layerDistance, numLayers, errorPhi):

    # The "tracer" vector points from the center to the origin;
    # it will trace the particle track
    tracer = rt.TVector2(-centerX, -centerY)
    # g holds the positions of the particle
    g = rt.TGraph(1000).Clone()
    # h holds the positions of the hits ~ intersections between the track and the layers
    h = rt.TGraphErrors(9).Clone()
    rng2 = rt.TRandom3(0)

    # Rotate the tracer to draw the track
    scalingFactor = 100 * tracer.Mod()
    for i in list(range(1000)):
        thisX = centerX + tracer.Rotate(sign * pi / scalingFactor * i).X()
        thisY = centerY + tracer.Rotate(sign * pi / scalingFactor * i).Y()
        g.SetPoint(i, thisX, thisY)

    # For each layer, find the closest point in the track to it
    # (i.e. the one that has the same radius)
    for j in list(range(numLayers)):
        targetRadius = layerDistance * (1 + j)
        best_k = -1
        best_distance = 9999.9
        for k in list(range(1000)):
            thisRadius = sqrt(pow(g.GetX()[k], 2) + pow(g.GetY()[k], 2))
            if abs(targetRadius - thisRadius) < best_distance:
                best_k = k
                best_distance = abs(targetRadius - thisRadius)

        # Found the best X and Y
        bestX = g.GetX()[best_k]
        bestY = g.GetY()[best_k]
        # Reuse the tracer here
        tracer.SetX(bestX)
        tracer.SetY(bestY)
        thisPhi = tracer.Phi()
        thisRadius = tracer.Mod()
        # Optional error in phi
        if errorPhi > 0.0:
            thisPhi = rng2.Gaus(thisPhi, errorPhi)
        tracer.SetMagPhi(thisRadius, thisPhi)
        # Propagate uncertainty to X and Y
        dx = thisRadius * sin(thisPhi) * errorPhi
        dy = thisRadius * cos(thisPhi) * errorPhi
        h.SetPoint(j, tracer.X(), tracer.Y())
        h.SetPointError(j, dx, dy)

    return g, h


def makeChargedParticleTracks():

    ### Setup the simulation
    layerDistance = 1.0  # cm
    particlePt = 0.9  # GeV
    # relDispersionPt = 0.1 # not used at the moment
    magneticField = 20.0  # Tesla
    charge = 1.0  # elementary charges
    numLayers = 10
    numTracks = 15
    errorPhi = 0.01
    paddingForGraph = 1.1
    curvConstant = 0.003  # for radius in cm, pt in GeV, q in elem. charge, B in T

    outFileName = (
        str(numTracks)
        + "Particles"
        + "_Pt"
        + str(particlePt).replace(".", "p")
        + "_BField"
        + str(int(magneticField))
        + "_errorPhi"
        + str(errorPhi).replace(".", "p")
        + ".txt"
    )
    print(outFileName)
    outFile = open(outFileName, "w")

    ### Calculate the curvature radius of the particle tracks
    curvRadius = particlePt / (curvConstant * charge * magneticField)
    print("curvRadius = ", curvRadius)

    ### We want to histogram all phis of the particles we simulated
    histoPhi = rt.TH1F("histoPhi", "histoPhi", 20, 0, pi / 2)

    ### Draw the detector layers
    cv = rt.TCanvas("detector", "detector", 600, 600)
    cv.Draw()
    distanceForGraph = ceil(layerDistance * numLayers * paddingForGraph * 10) / 10.0
    histo = rt.TH1F("histo", "histo", 1, -distanceForGraph, distanceForGraph)
    histo.SetLineWidth(0)
    histo.Draw()
    histo.GetXaxis().SetRangeUser(-distanceForGraph, distanceForGraph)
    histo.GetYaxis().SetRangeUser(-distanceForGraph, distanceForGraph)
    histo.Draw()
    for i in list(range(numLayers)):
        thisRadius = (numLayers - i) * layerDistance
        print(thisRadius)
        e = rt.TEllipse(0.0, 0.0, thisRadius).Clone()
        e.SetFillStyle(0)
        e.SetLineColor(rt.kGray + 1)
        e.Draw("SAME")
    cv.Draw("SAME")

    rng = rt.TRandom3(0)
    for nt in list(range(numTracks)):
        centerX = ctypes.c_double(0.0)
        centerY = ctypes.c_double(0.0)
        ### Not used at the moment
        # R = rng.Gaus(curvRadius, relDispersionPt)
        sign = 0

        ### Tracks anywhere
        # rng.Circle(centerX, centerY, R)
        # sign = rng.Uniform(-0.5,0.5) > 0 ? 1 : -1
        ### Tracks in first quadrant only
        while centerX.value * centerY.value >= 0.0:
            rng.Circle(centerX, centerY, curvRadius)
            if centerX.value < 0:
                sign = 1
            elif centerX.value > 0:
                sign = -1

        # It is safe to use regular ATan instead of ATan2 because
        # we know we are in the first quadrant only
        truePhi = atan(-centerX.value / centerY.value)

        ### Make the trajectory (all positions of the particle)
        ### and the track (set of hits in the detector)
        thisTrajectory, thisTrack = makeTrack(
            centerX.value, centerY.value, sign, layerDistance, numLayers, errorPhi
        )

        ### Should we draw this track?
        drawThisTrack = rng.Uniform(0, 1) < 0.01  # 1% chance
        if drawThisTrack:
            thisTrajectory.SetLineColor(rt.kBlue)
            thisTrajectory.SetLineStyle(rt.kDashed)
            thisTrajectory.Draw("C")
            thisTrack.SetMarkerSize(0.6)
            thisTrack.Draw("P")

        ### Write the track in the outFile
        thisLine = ""
        thisLine += str(truePhi) + "\t"
        for np in list(range(thisTrack.GetN())):
            thisLine += (
                str(thisTrack.GetX()[np])
                + "\t"
                + str(thisTrack.GetY()[np])
                + "\t"
                + str(thisTrack.GetEX()[np])
                + "\t"
                + str(thisTrack.GetEY()[np])
                + "\t"
            )
        thisLine += "\n"
        outFile.write(thisLine)

        histoPhi.Fill(truePhi)

    # Draw the detector and the phi histogram
    cv.SaveAs("detector.pdf")
    histoPhi.Draw()
    histoPhi.GetYaxis().SetRangeUser(0, 2 * histoPhi.GetMaximum())
    cv.SaveAs("histogramPhi.pdf")
    outFile.close()
    return


def main():
    makeChargedParticleTracks()


if __name__ == "__main__":
    main()