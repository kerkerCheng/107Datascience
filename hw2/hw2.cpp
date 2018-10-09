#include <iostream>
#include <cstdio>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <utility>
#include <algorithm>
#include <map>
#include "fp_tree.hpp"


using namespace std;

void print_trans(vector< vector<int> >& trans);
void print_pairs(vector< pair<int, int> >& one_item_set);
void print_tree_branch(Tree *tr);
void print_map_sizes(map<int, std::vector<Node *> >& header_table);


Node::Node(int tag)
{
	_tag = tag;
	_freq = 1;
}


Node::Node()
{
	_tag = -1;
	_freq = -1;
}


Tree::Tree(vector< vector<int> >& trans)
{
	root = new Node();

	for(vector< vector<int> >::iterator it1 = trans.begin(); it1 != trans.end(); it1++)
	{
		Node *current = root;
		for(vector<int>::iterator it2 = it1->begin(); it2 != it1->end(); it2++)
		{
			bool if_find = false;
			if(!(current->children).empty())
			{
				for(vector<Node *>::iterator it = (current->children).begin(); it != (current->children).end(); it++)
				{
					if((*it)->_tag == *it2)
					{
						if_find = true;
						(*it)->_freq ++;
						current = *it;
						break;
					}
				}
			}
			if(!if_find)
			{
				(current->children).push_back(new Node());
				((current->children).back())->_tag = *it2;
				((current->children).back())->_freq = 1;
				((current->children).back())->parent = current;
				current = (current->children).back();

				//Header table construction
				header_table[*it2].push_back(current);
			}
		}
	}
}


int main(void)
{
	//redirect stdin to 1.in file
	freopen("sample2.in", "r", stdin);
	vector< vector<int> > transactions;
	vector< pair<int, int> > one_item_set(1000);
	string line;

	//Initialize the one_item_set vector
	for(int i=0; i<1000; i++)
		one_item_set[i] = make_pair(0, i);

	//read until EOF
	while(!getline(cin, line).eof())
	{
		vector<int> arr;
		istringstream ssline(line);
		string number;
		while(getline(ssline, number, ','))
		{
			int item = atoi(number.c_str());
			arr.push_back(item);
			(one_item_set[item]).first += 1;
		}

		transactions.push_back(arr);
	}

	//Get 1-item ordered set
	int num_of_trans = transactions.size();
	sort(one_item_set.begin(), one_item_set.end());
	reverse(one_item_set.begin(), one_item_set.end());

	//Erase items that absence
	for(int i=0; i<one_item_set.size(); i++)
	{
		if((one_item_set[i]).first == 0)
		{
			one_item_set.erase(one_item_set.begin()+i, one_item_set.end());
			break;
		}
	}

	//Create ordered frequent 1-item vector
	vector< vector<int> > re_transactions(num_of_trans);
	for(vector< pair<int, int> >::iterator it1 = one_item_set.begin(); it1 != one_item_set.end(); it1++)
	{
		int trans_ind = 0;
		for(vector< vector<int> >::iterator it2 = transactions.begin(); it2 != transactions.end(); it2++)
		{
			if(find(it2->begin(), it2->end(), it1->second) != it2->end())
				re_transactions[trans_ind].push_back(it1->second);

			trans_ind += 1;
		}
	}

	Tree a = Tree(re_transactions);
	print_tree_branch(&a);

	// print_trans(transactions);
	// print_pairs(one_item_set);
	// print_trans(re_transactions);
	print_map_sizes(a.header_table);



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

void print_pairs(vector< pair<int, int> >& one_item_set)
{
	for(vector< pair<int, int> >::iterator it = one_item_set.begin(); it != one_item_set.end(); it++)
		cout << it->second << "的次數 = " << it->first << endl;
}

void print_tree_branch(Tree *tr)
{
	Node *n = tr->root;

	while(!(n->children).empty())
	{
		cout << n->_tag << ", 次數: " << n->_freq << endl;
		cout << n->children.size() << endl;
		n = (n->children)[0];
	}
}

void print_map_sizes(map<int, std::vector<Node *> >& header_table)
{
	for(map<int, std::vector<Node *> >::iterator it = header_table.begin(); it != header_table.end(); it++)
		cout << it->first << " , vector_size = " << (it->second).size() << endl;
}
