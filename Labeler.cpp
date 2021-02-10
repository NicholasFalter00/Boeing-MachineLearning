/*
This program is meant to add a label to the logs, line by line
The label is provided as user input
*/
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

string formatLabel(string label);
void logLabeler(ifstream infile, string finalFile, string label);

int main(){

    //string fileName = "test.txt", label;
    string fileName = "firstFile.txt", label;
    ifstream infile(fileName.c_str(), ios::in); // Open file to read

    if (infile.good()){ // Check if the file exists and can be read
        cout << "Enter label: "; // Grab label to append to each line
        cin >> label;

        label = formatLabel(label); // Replace spaces in label w/ "_"

        //logLabeler(infile, "final.txt", label); // Add label to beginning of each log
        ofstream outfile("final.txt", ios::out); // Open file to write
        string line;
        int lineCounter = 0;

        while (getline(infile, line)){ // Check & store next line from infile
            outfile << (label + " " + line + "\n"); // Write to file
            lineCounter++; // Count the number of logs labeled
        }
        cout << lineCounter + " logs have been successfully labeled" << endl;

        outfile.close(); // Close opened ofstream file
    } else
        cout << "!!!ERROR: Couldn't open test.txt!!!" << endl; // *Did not open file*

    infile.close(); // Close opened ifstream file
    return 0;
}

// Replace spaces in label w/ "_"
string formatLabel(string label){
    int labelSize = label.length();
    for(int i = 0; i < labelSize; i++)
        if(label.at(i) == ' ')
            label.at(i) = '_';

    return label;
}

// Add label to beginning of each log and get number of logs labeled
/*void logLabeler(ifstream infile, string finalFile, string label){
    ofstream outfile(finalFile, ios::out); // Open file to write
    string line;
    int lineCounter = 0;

    while (getline(infile, line)){ // Check & store next line from infile
        outfile << (label + " " + line + "\n"); // Write to file
        lineCounter++; // Count the number of logs labeled
    }
    cout << lineCounter + " logs have been successfully labeled" << endl;

    outfile.close(); // Close opened ofstream file
}*/
