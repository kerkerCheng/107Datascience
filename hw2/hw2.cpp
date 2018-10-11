#include <iostream>
#include <cstdio>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <utility>
#include <algorithm>
#include <map>
#include "fp_tree.hpp"
#define MAX_FREQ	1000000

using namespace std;

int min_sup = 200;

void print_trans(vector< vector<int> >& trans);
void print_pairs(vector< pair<int, int> >& one_item_set);
void print_tree_branch(Tree *tr);
void print_map_sizes(map<int, std::vector<Node *> >& header_table);
void print_cond_pattern(vector<pair<vector<int>, int> >& cond_pat);
void get_cond_pattern(vector<pair<vector<int>, int> >& cond_pat, int target, Tree *tr);
void build_cond_pat_trans(vector<pair<vector<int>, int> >& cond_pat, vector<vector<int> >& trans, vector< pair<int, int> >& one_item_set);
void build_cond_fptr_retransaction(int tag, Tree *tr, vector< vector<int> > &re_transactions);
void tree_mining(Tree *tr, vector< pair<int, int> >& one_item_set, vector<pair<vector<int>, int> >& freq_items, vector<int>& prefix);


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

	//Erase items that lower than min_sup
	for(int i=0; i<one_item_set.size(); i++)
	{
		if((one_item_set[i]).first < min_sup)
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

	Tree fp = Tree(re_transactions);

	// print_tree_branch(&a);
	// print_trans(transactions);
	// print_pairs(one_item_set);
	// print_trans(re_transactions);
	// print_map_sizes(a.header_table);
	// print_cond_pattern(cond_pat);

	//Mining FP-tree
	vector<pair<vector<int>, int> > freq_items;
	vector<int> prefix;
	tree_mining(&fp, one_item_set, freq_items, prefix);



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

void print_cond_pattern(vector<pair<vector<int>, int> >& cond_pat)
{
	for(vector<pair<vector<int>, int> >::iterator it1 = cond_pat.begin(); it1 != cond_pat.end(); it1++)
	{
		for(vector<int>::iterator it2 = (it1->first).begin(); it2 != (it1->first).end(); it2++)
		{
			if(it2 != (it1->first).end()-1)
				cout << *it2 << ",";
			else
				cout << *it2 << " : ";
		}

		cout << it1->second << endl;
	}
}

void get_cond_pattern(vector<pair<vector<int>, int> >& cond_pat, int target, Tree *tr)
{
	for(vector<Node *>::iterator it1 = (tr->header_table)[target].begin(); it1 != (tr->header_table)[target].end(); it1++)
	{
		pair< vector<int>, int> path;
		Node *current = *it1;
		path.second = current->_freq;

		while(current->_freq != -1)
		{
			current = current->parent;
			if(current->_freq != -1)
				(path.first).push_back(current->_tag);
		}

		reverse((path.first).begin(), (path.first).end());
		cond_pat.push_back(path);
	}
}

void build_cond_pat_trans(vector<pair<vector<int>, int> >& cond_pat, vector<vector<int> >& trans, vector< pair<int, int> >& one_item_set)
{
	for(vector<pair<vector<int>, int> >::iterator it1 = cond_pat.begin(); it1 != cond_pat.end(); it1++)
	{
		vector<int> path;
		for(vector<int>::iterator it2 = (it1->first).begin(); it2 != (it1->first).end(); it2++)
		{
			path.push_back(*it2);
			one_item_set.at(*it2).first += it1->second;
		}
		for(int i=0; i < it1->second; i++)
			trans.push_back(path);
	}
}

void build_cond_fptr_retransaction(int tag, Tree *tr, vector< vector<int> > &re_transactions, vector< pair<int, int> >& one_item_set)
{
	vector<pair<vector<int>, int> > cond_pat;
	get_cond_pattern(cond_pat, tag, tr);

	int num_of_trans = 0;
	for(vector<pair<vector<int>, int> >::iterator it = cond_pat.begin(); it != cond_pat.end(); it++)
		num_of_trans += it->second;


	one_item_set.resize(1000);
	for(int i=0; i<1000; i++)
	{
		one_item_set[i].first = 0;
		one_item_set[i].second = i;
	}


	vector< vector<int> > cond_pat_trans;
	build_cond_pat_trans(cond_pat, cond_pat_trans, one_item_set);

	sort(one_item_set.begin(), one_item_set.end());
	reverse(one_item_set.begin(), one_item_set.end());

	//Erase items that lower than min_sup
	for(int i=0; i<one_item_set.size(); i++)
	{
		if((one_item_set[i]).first < min_sup)
		{
			one_item_set.erase(one_item_set.begin()+i, one_item_set.end());
			break;
		}
	}

	re_transactions.resize(num_of_trans);

	for(vector< pair<int, int> >::iterator it1 = one_item_set.begin(); it1 != one_item_set.end(); it1++)
	{
		int trans_ind = 0;
		for(vector< vector<int> >::iterator it2 = cond_pat_trans.begin(); it2 != cond_pat_trans.end(); it2++)
		{
			if(find(it2->begin(), it2->end(), it1->second) != it2->end())
				re_transactions[trans_ind].push_back(it1->second);

			trans_ind += 1;
		}
	}
}

void tree_mining(Tree *tr, vector< pair<int, int> >& one_item_set, vector<pair<vector<int>, int> >& freq_items, vector<int>& prefix)
{
	for(vector< pair<int, int> >::reverse_iterator it1 = one_item_set.rbegin(); it1 != one_item_set.rend(); it1++)
	{
		vector<int> new_set(prefix);
		new_set.push_back(it1->second);

		vector< vector<int> > re_transactions;
		vector< pair<int, int> > one_item_set_ss;
		build_cond_fptr_retransaction(it1->second, tr, re_transactions, one_item_set_ss);

		// print_pairs(one_item_set_ss);

		Tree t = Tree(re_transactions);
		freq_items.push_back(make_pair(new_set, it1->first));

		for(vector<int>::iterator it3 = new_set.begin(); it3 != new_set.end(); it3++)
			cout << *it3 << ",";
		cout << "次數: " << it1->first << endl;

		if(((t.root)->children).size() != 0)
			tree_mining(&t, one_item_set_ss, freq_items, new_set);
		
	}
}

void print_freq_sets(vector<pair<vector<int>, int> >& freq_items)
{
	for(vector<pair<vector<int>, int> >::iterator it = freq_items.begin(); it != freq_items.end(); it++)
	{
		for(vector<int>::iterator it2 = (it->first).begin(); it2 != (it->first).end(); it2++)
		{
			if(it2 != (it->first).end()-1)
				cout << *it2 << ",";
			else
				cout << *it2;
		}

		cout << "    : " << it->second << endl;
	}
}
