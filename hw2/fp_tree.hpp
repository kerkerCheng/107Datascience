#include <iostream>
#include <cstdio>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <utility>
#include <algorithm>
#include <map>

class Node;
class Tree;

class Node
{
public:
	int _tag;
	int _freq;
	Node *parent;
	std::vector<Node *> children;

	Node(int);
	Node();
};

class Tree
{
public:
	Node* root;
	std::map<int, std::vector<Node *> > header_table;
	Tree(std::vector< std::vector<int> >&);
	Tree(std::vector< std::pair< std::vector<int>, int> >&);

};
