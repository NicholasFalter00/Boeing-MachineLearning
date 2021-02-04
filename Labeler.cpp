/*
This program is meant to add a label to the logs, line by line
The label is provided as user input
*/
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

int main(){

    string fileName = "test.txt", label, line;
    ifstream infile(fileName.c_str(), ios::in); // Open file to read

    if (infile.good()){ // Check if the file exists and can be read
        ofstream outfile("final.txt", ios::out); // Open file to write

        cout << "Enter label: "; // Grab label to append to each line
        cin >> label;

        while (getline(infile, line)) // Check & store next line from infile
            outfile << (line + " " + label + "\n"); // Write to file

        outfile.close(); // Close opened ofstream file
    } else
        cout << "!!!ERROR: Couldn't open test.txt!!!" << endl; // *Did not open file*

    infile.close(); // Close opened ifstream file
    return 0;
}
