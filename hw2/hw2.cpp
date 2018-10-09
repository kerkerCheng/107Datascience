#include <iostream>
#include <cstdio>
#include <sstream>
#include <vector>
#include <stdlib.h>

using namespace std;

void print_trans(vector< vector<int> >& trans);


int main(void)
{
	//redirect stdin to 1.in file
	freopen("sample2.in", "r", stdin);
	vector< vector<int> > transactions;
	string line;
	//read until EOF
	while(!getline(cin, line).eof())
	{
		vector<int> arr;
		istringstream ssline(line);
		string number;
		while(getline(ssline, number, ','))
		arr.push_back(atoi(number.c_str()));
		transactions.push_back(arr);
	}


	

	// print_trans(transactions);

	return 0;
}

void print_trans(vector< vector<int> >& trans)
{
	for(vector< vector<int> >::iterator it1 = trans.begin(); it1 != trans.end(); it1++)
	{
		for(vector<int>::iterator it2 = (*it1).begin(); it2 != (*it1).end(); it2++)
		{
			if(it2 == (*it1).end()-1)
			{
				cout << (*it2) << endl;
				break;
			}
			cout << *it2 << ",";
		}
	}
}