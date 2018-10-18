#include <iostream>
#include <cstdio>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <utility>
#include <algorithm>
#include <map>
#include <queue>
#include <math.h>
#include "fp_tree.hpp"
#define MAX_FREQ	1000000

using namespace std;

int min_sup;
int num_of_trans;
double min_sup_d;
double num_of_trans_d;

struct Comp{
    bool operator()(const pair<vector<int>, int>& a, const pair<vector<int>, int>& b){
        if(a.first.size() > b.first.size())
        	return true;
        else if(a.first.size() < b.first.size())
        	return false;
        else if(a.first.size() == b.first.size())
        {
        	for(int i=0; i<a.first.size(); i++)
        	{
        		if((a.first).at(i) > (b.first).at(i))
        			return true;
        		else if((a.first).at(i) < (b.first).at(i))
        			return false;
        	}
        }
        return false;
    }
};


void print_trans(vector< vector<int> >& trans);
void print_pairs(vector< pair<int, int> >& one_item_set);
void print_tree_branch(Tree *tr);
void print_map_sizes(map<int, std::vector<Node *> >& header_table);
void print_cond_pattern(vector<pair<vector<int>, int> >& cond_pat);
void get_cond_pattern(vector<pair<vector<int>, int> >& cond_pat, int target, Tree *tr);
void build_cond_pat_trans(vector<pair<vector<int>, int> >& cond_pat, vector<vector<int> >& trans, vector< pair<int, int> >& one_item_set);
void build_cond_fptr_retransaction(int tag, Tree *tr, vector< vector<int> > &re_transactions);
void tree_mining(Tree *tr, vector< pair<int, int> >& one_item_set, priority_queue<pair<vector<int>, int>, vector<pair<vector<int>, int> >, Comp>& freq_items, vector<int>& prefix);
void print_freq_sets(priority_queue<pair<vector<int>, int>, vector<pair<vector<int>, int> >, Comp>& freq_items);
void cond_pat_cleaning(vector<pair<vector<int>, int> >& cond_pat, vector< pair<int, int> >& one_item_set, int min_sup);
void output_freq_sets(priority_queue<pair<vector<int>, int>, vector<pair<vector<int>, int> >, Comp>& freq_items, string output_path);
void cond_pat_cleaning_one_item_set(vector<pair<vector<int>, int> >& cond_pat, vector< pair<int, int> >& one_item_set, int min_sup);
void tree_mining2(Tree *tr, vector< pair<int, int> >& one_item_set, priority_queue<pair<vector<int>, int>, vector<pair<vector<int>, int> >, Comp>& freq_items);
vector< vector< pair<int, int> > > all_subsets(vector< pair<int, int> > items);

struct MatchTag
{
	 MatchTag(const int& i) : i_(i) {}
	 bool operator()(const Node* obj)
	 {
	   return (obj->_tag) == i_;
	 }
   int i_;
};

struct MatchPairFirst
{
	 MatchPairFirst(const int& i) : i_(i) {}
	 bool operator()(const pair<int, int>& obj) const
	 {
	   return obj.first == i_;
	 }
 private:
   const int i_;
};


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

Tree::Tree(vector<pair<vector<int>, int> >& cond_pat)
{
	root = new Node();

	for(vector<pair<vector<int>, int> >::iterator it1 = cond_pat.begin(); it1 != cond_pat.end(); it1++)
	{
		Node *current = root;
		for(vector<int>::iterator it2 = it1->first.begin(); it2 != it1->first.end(); it2++)
		{
			bool if_find = false;
			if(!(current->children).empty())
			{
				for(vector<Node *>::iterator itf = (current->children).begin(); itf != (current->children).end(); itf++)		
				{
					if((*itf)->_tag == *it2)
					{
						if_find = true;
						(*itf)->_freq += it1->second;
						current = *itf;
						break;
					}
				}
			}
			if(!if_find)
			{
				(current->children).push_back(new Node());
				((current->children).back())->_tag = *it2;
				((current->children).back())->_freq = it1->second;
				((current->children).back())->parent = current;
				current = (current->children).back();

				//Header table construction
				header_table[*it2].push_back(current);
			}
		}
	}
}


int main(int argc, char *argv[])
{
	string input_file = argv[2];
	string output_path = argv[3];
	min_sup_d = atof(argv[1]);

	cout << "output_file_path : " << output_path << endl;
	cout << "min_sup_d : " << min_sup_d << endl;

	//redirect stdin to 1.in file
	freopen(input_file.c_str(), "r", stdin);
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
	num_of_trans = transactions.size();
	num_of_trans_d = (double)num_of_trans;
	min_sup = (int)(min_sup_d*num_of_trans_d);
	cout << "min_sup : " << min_sup << endl;
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
	// vector<pair<vector<int>, int> > freq_items;
	priority_queue<pair<vector<int>, int>, vector<pair<vector<int>, int> >, Comp> freq_items;
	vector<int> prefix;
	// tree_mining(&fp, one_item_set, freq_items, prefix);
	tree_mining2(&fp, one_item_set, freq_items);
	cout << freq_items.size() << endl;
	output_freq_sets(freq_items, output_path);

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

void cond_pat_cleaning(vector<pair<vector<int>, int> >& cond_pat, vector< pair<int, int> >& one_item_set, int min_sup)
{
	one_item_set.resize(1000);
	for(int i=0; i<1000; i++)
	{
		one_item_set[i].first = 0;
		one_item_set[i].second = i;
	}

	for(vector<pair<vector<int>, int> >::iterator it = cond_pat.begin(); it != cond_pat.end(); it++)
	{
		for(vector<int>::iterator it2 = it->first.begin(); it2 != it->first.end(); it2++)
			one_item_set.at(*it2).first += it->second;
	}
	
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

	vector<pair<vector<int>, int> > new_cond_pat;


	for(vector<pair<vector<int>, int> >::iterator it = cond_pat.begin(); it != cond_pat.end(); it++)
	{
		vector<int> tmp;
		for(vector<int>::iterator it2 = it->first.begin(); it2 != it->first.end(); it2++)
		{
			for(vector< pair<int, int> >::iterator it3 = one_item_set.begin(); it3 != one_item_set.end(); it3++)
			{
				if(it3->second == *it2)
				{
					tmp.push_back(*it2);
					break;
				}
			}
		}

		new_cond_pat.push_back(make_pair(tmp, it->second));
	}

	cond_pat = new_cond_pat;
}

void cond_pat_cleaning_one_item_set(vector<pair<vector<int>, int> >& cond_pat, vector< pair<int, int> >& one_item_set, int min_sup)
{
	one_item_set.resize(1000);
	for(int i=0; i<1000; i++)
	{
		one_item_set[i].first = 0;
		one_item_set[i].second = i;
	}

	for(vector<pair<vector<int>, int> >::iterator it = cond_pat.begin(); it != cond_pat.end(); it++)
	{
		for(vector<int>::iterator it2 = it->first.begin(); it2 != it->first.end(); it2++)
			one_item_set.at(*it2).first += it->second;
	}
	
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
}

void tree_mining(Tree *tr, vector< pair<int, int> >& one_item_set, priority_queue<pair<vector<int>, int>, vector<pair<vector<int>, int> >, Comp>& freq_items, vector<int>& prefix)
{
	for(vector< pair<int, int> >::reverse_iterator it1 = one_item_set.rbegin(); it1 != one_item_set.rend(); it1++)
	{
		vector<int> new_set(prefix);
		new_set.push_back(it1->second);

		vector< vector<int> > re_transactions;
		vector< pair<int, int> > new_one_item_set;
		vector<pair<vector<int>, int> > cond_pat;

		get_cond_pattern(cond_pat, it1->second, tr);
		cond_pat_cleaning(cond_pat, new_one_item_set, min_sup);
		Tree t = Tree(cond_pat);


		// build_cond_fptr_retransaction(it1->second, tr, re_transactions, new_one_item_set);
		// print_pairs(new_one_item_set);
		// Tree t = Tree(re_transactions);

		sort(new_set.begin(), new_set.end());
		freq_items.push(make_pair(new_set, it1->first));

		// for(vector<int>::iterator it3 = new_set.begin(); it3 != new_set.end(); it3++)
		// 	cout << *it3 << ",";
		// cout << "次數: " << it1->first << endl;

		if(((t.root)->children).size() != 0)
			tree_mining(&t, new_one_item_set, freq_items, new_set);
		
	}
}

void print_freq_sets(priority_queue<pair<vector<int>, int>, vector<pair<vector<int>, int> >, Comp>& freq_items)
{
	while(!freq_items.empty())
	{
		pair<vector<int>, int> pp= freq_items.top();
		for(vector<int>::iterator it = pp.first.begin(); it != pp.first.end(); it++)
		{
			if(it != pp.first.end()-1)
				cout << *it << ",";
			else
				cout << *it;
		}
		cout << ":" << freq_items.top().second << endl;

		freq_items.pop();
	}
}

void tree_mining2(Tree *tr, vector< pair<int, int> >& one_item_set, priority_queue<pair<vector<int>, int>, vector<pair<vector<int>, int> >, Comp>& freq_items)
{
	for(vector< pair<int, int> >::reverse_iterator it1 = one_item_set.rbegin(); it1 != one_item_set.rend(); it1++)
	{
		vector< vector<int> > re_transactions;
		vector< pair<int, int> > one_item_set_x;
		vector<pair<vector<int>, int> > cond_pat;

		get_cond_pattern(cond_pat, it1->second, tr);
		cond_pat_cleaning_one_item_set(cond_pat, one_item_set_x, min_sup);

		vector< vector< pair<int, int> > > tmp = all_subsets(one_item_set_x);

		cout << tmp.size() << endl;


		if(tmp.size() == 0)
		{
			cout << "itit" << it1->first << endl;
			if(it1->first > min_sup)
			{
				vector<int> sets;
				sets.push_back(it1->second);
				freq_items.push(make_pair(sets, it1->first));
			}

			continue;
		}

		for(vector< vector< pair<int, int> > >::iterator it2 = tmp.begin(); it2 != tmp.end(); it2++)
		{
			int min = 1000000;
			vector<int> sets;
			for(vector< pair<int, int> >::iterator it3 = it2->begin(); it3 != it2->end(); it3++)
			{
				sets.push_back(it3->second);
				if(it3->first < min)
					min = it3->first;
			}
			if(it1->first < min)
				min = it1->first;

			sets.push_back(it1->second);
			sort(sets.begin(), sets.end());
			
			freq_items.push(make_pair(sets, min));
		}
	}
}

vector< vector< pair<int, int> > > all_subsets(vector< pair<int, int> > items)
{
	if(items.size() == 1)
	{
		vector< pair<int, int> > blank;
		vector< vector< pair<int, int> > > re;
		re.push_back(items);
		re.push_back(blank);
		return re;
	}
	else if(items.size() >= 1)
	{
		pair<int, int> first(items[0]);
		vector< pair<int, int> >::const_iterator fir = items.begin()+1;
		vector< pair<int, int> >::const_iterator last = items.end();
		vector< pair<int, int> > in(fir, last);

		vector< vector< pair<int, int> > > tmp = all_subsets(in);

		vector< vector< pair<int, int> > > tmp2(tmp);
		for(int i=0; i < tmp2.size(); i++)
			tmp2[i].push_back(first);
		
		tmp.insert(tmp.end(), tmp2.begin(), tmp2.end());
		return tmp;
	}
	else
	{
		vector< vector< pair<int, int> > > s;
		return s;
	}
}


void output_freq_sets(priority_queue<pair<vector<int>, int>, vector<pair<vector<int>, int> >, Comp>& freq_items, string output_path)
{
	FILE *p;
	p = fopen(output_path.c_str(), "w+");

	while(!freq_items.empty())
	{
		pair<vector<int>, int> pp= freq_items.top();
		for(vector<int>::iterator it = pp.first.begin(); it != pp.first.end(); it++)
		{
			if(it != pp.first.end()-1)
				fprintf(p, "%d,", *it);
			else
				fprintf(p, "%d", *it);
		}
		double f(freq_items.top().second);
		double sup_percent = round((f/num_of_trans_d)*10000) / 10000;
		fprintf(p, ":%.4f\n", sup_percent);

		freq_items.pop();
	}
}
