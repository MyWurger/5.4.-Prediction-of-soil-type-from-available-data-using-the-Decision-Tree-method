#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <unordered_set>

using namespace std;

struct Vector
{
    vector<double> values; // ������ ��������
    double humus;          // �������� ������
};

// ���������� �������� ������� �����
//������� entropy ������������ �������� ��� ��������� ������ �����, ��� �������� �������� ������� ���������������� ��� ������������� ����� � ������.
double entropy(const vector<double>& labels)
{
    // �������� ������ ���������� ����� �� ��������� ������ labels.
    unordered_set<double> unique_labels(labels.begin(), labels.end());  // ��������������� ��������� 

    double entropy = 0.0;
    for (const auto& label : unique_labels) // ���� �� ���� ���������� ������ � unique_labels
    {
        double count_1 = count(labels.begin(), labels.end(), label); // ���������� count_1 ���������������� ����������� ��������� ������� ����� � �������� ������ labels � ������� ������� count().
        double p = count_1 / labels.size();                          // ����������� ��������� ������� �����
        entropy -= p * log2(p);                                      // ��� ��������� ��������� ����� ������ ���������� ����� � ����� ���� ���������������� �������.
    }
    return entropy;
}


// ���������� ��������������� �������� �� ���������� ������� (�������� ����������)
// ��������� ������� ���������� ��� ��������� ������ �� ��� ����� �� ������������� �������� (feature).
double information_gain(const vector<Vector>& data, int feature)
{
    double total_entropy = entropy(vector<double>(data.size(), 0.0)); // ������ �������� (���� ����������������) ��� ����� ������ ������.
    vector<double> values(data.size());                               // ������ �������� ��������� feature ��� ������� ������� � ������ ������ data

    for (int i = 0; i < data.size(); i++)
    {
        values[i] = data[i].values[feature];
    }

    sort(values.begin(), values.end());   // ���������� �������� �� ����������� 
    double best_gain = INT_MIN;           // ���������� ��� �������� �������� ������������� �������� ���������� 
    double best_split_value = 0.0;        // ���������� ��� �������� ��������, �� �������� ����� ����������� ��������� 


    // ���� �� ���� ��������� ������� values, ����� ���������� ��������, �.�. ��������� �� ��� ����� �������� ������ ��� ������� ���� �� ���� ��������� ��������
    for (int i = 0; i < data.size() - 1; i++)
    {
        double split_value = (values[i] + values[i + 1]) / 2.0; // ������� �������� ����� ������� � ��������� ��������� � values
        vector<double> left_labels, right_labels;               // �������, � ������� ����� �������� ����� ��������, ��������������� ����� � ������ ������ ���������

        for (const auto& vector : data)               // � ����� �� ���� �������� � data �������������� �������� �������� �������� feature ��� ������� �������
        {
            if (vector.values[feature] < split_value) // ���� �������� ������ split_value, �� ����� ����������� � ����� ���������, 
            {
                left_labels.push_back(vector.humus);
            }

            else //����� - � ������ ���������.
            {
                right_labels.push_back(vector.humus);
            }
        }

        double left_entropy = entropy(left_labels);   // ���������� �������� ������ ���������
        double right_entropy = entropy(right_labels); // ���������� �������� ������� ���������

        // ���������� �������� ���������� ��� ������� ����� total_entropy � ��������� ������� ����� � ������ ����� ��������� � ����� ��������
        double gain = total_entropy - (left_labels.size() * left_entropy + right_labels.size() * right_entropy) / data.size();

        if (gain > best_gain) // ���� ������� ���������� ������ �������� ��������, ��
        {
            best_gain = gain;
            best_split_value = split_value;
        }
    }

    return best_gain; // ����������� ���������� �������� ����������, ������� ������� ����� ��� ��������� ������ �� �������� feature
}


// ������� ���������� ���������� �������� ��� ���������� ������ �� ������ ���������� ����������
int find_best_feature(const vector<Vector>& data)
{
    int p = data[0].values.size(); // ����������� ���-�� ��������� ������� ������� 
    double best_gain = INT_MIN;    // ���������� ��� �������� �������� ������������� �������� ���������� 
    int best_feature = -1;         // ���������� ���������� ��������

    for (int j = 0; j < p; j++) // ���� �� ���� ���������
    {
        double gain = information_gain(data, j); // ���������� ������� information_gain ��� ���������� ��������������� �������� ��� �������� j � ������ data

        if (gain > best_gain)  // ���� �������������� ������� gain ������ �������� ���������� �������� best_gain, ��
        {
            best_gain = gain;
            best_feature = j;
        }
    }

    return best_feature; // ����������� �������� � ��������� �������������� ���������
}

// ��������� ��� ���������� ������ �������
struct Node
{
    int feature;        // ���������� ��� �������� ��������
    double split_value; // ���������� ��� �������� �������� �������� 
    double prediction;  // ���������� ��� �������� �������������� ��������
    Node* left;         // ��������� �� ������ �������
    Node* right;        // ��������� �� ������� �������
};


// ������� ���������� ������ ������� �� ������ ������ ������ � ������� ������
Node* build_tree(const vector<Vector>& data, int depth)
{
    if (depth == 0 || data.size() < 2) // ���� ������� ����� 0 ��� ������ ������ ������ 2, �� 
    {
        double prediction = 0.0; // ����������, ������� ������ ������������� ��������

        for (const auto& vector : data)
        {
            prediction = prediction + vector.humus;
        }

        prediction = prediction / data.size();                          // ���������� �������� �������� humus ��� ������� �������� ������� data
        Node* node = new Node{ -1, 0.0, prediction, nullptr, nullptr }; // �������� ������ ����

        return node; // ����������� ��������� �� ����� ����
    }

    int best_feature = find_best_feature(data); //���������� ������� find_best_feature, ������� ������� ��������� ������� (best_feature) ��� ���������� ������

    if (best_feature == -1)  // ���� �� ����� ���������� �������
    {
        double prediction = 0.0; // ����������, ������� ������ ������������� ��������

        for (const auto& vector : data)
        {
            prediction = prediction + vector.humus;
        }

        prediction = prediction / data.size();                          // ���������� �������� �������� humus ��� ������� �������� ������� data
        Node* node = new Node{ -1, 0.0, prediction, nullptr, nullptr }; // �������� ������ ����

        return node; // ����������� ��������� �� ����� ����                                                    
    }

    vector<double> values(data.size());

    for (int i = 0; i < data.size(); i++) // � ������ values ������������ �������� ���������������� �������� best_feature ��� ������� �������� ������� data
    {
        values[i] = data[i].values[best_feature];
    }

    sort(values.begin(), values.end()); // ���������� ������� �� �����������

    double split_value = (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2.0; // ���������� �������� ��� ���������� ������ (split_value)

    vector<Vector> left_data, right_data;

    for (const auto& vector : data) // ���� �� ������� �������� ������� data
    {
        if (vector.values[best_feature] < split_value) // ���� �������� ������ split_value, �� ������ ����������� � ����� ���������,
        {
            left_data.push_back(vector);
        }
        else // ����� � ������ ���������
        {
            right_data.push_back(vector);
        }
    }

    Node* node = new Node{ best_feature, split_value, 0.0, nullptr, nullptr }; // �������� ����� ����

    node->left = build_tree(left_data, depth - 1);   // ������� build_tree ��� ���������� ������ ��������� �� ������ ������ left_data � ������� depth - 1
    node->right = build_tree(right_data, depth - 1); // ������� build_tree ��� ���������� ������� ��������� �� ������ ������ right_data � ������� depth - 1

    return node; // ������������ ��������� �� ��������� ���� node
}



// ��������������� �������� ������ ��� ��������� �������, ��������� ���������� ������ �������� �������
// � ���������� ���������� ������� predict ����� ���������� ������������, ���������� ��� ��������� ������ ������ �� ������ ���������� ������ �������� �������.
double predict(Node* node, const vector<double>& test_data)
{
    if (node->left == nullptr && node->right == nullptr) // ���� � ���� ��� ��������, �� 
    {
        return node->prediction;  // ���������� ������������� ��������
    }

    //  ���� �������� �������� ������ �� �������� node->feature ������ �������� split_value �������� ����, ��
    if (test_data[node->feature] < node->split_value)
    {
        return predict(node->left, test_data); // ����������� ����� � ����� �������� �����
    }

    else
    {
        return predict(node->right, test_data);  // ����������� ����� � ������ �������� �����
    }
}



int main() {
    // Load the data from the file
    ifstream file;
    file.open("data.txt");
    if (!file.is_open())
    {
        return -1;
    }
    int n = 70;
    int m = 8;
    vector<Vector> data(n);

    for (int i = 0; i < n; i++)
    {
        data[i].values.resize(m);

        for (int j = 0; j < m; j++)
        {
            file >> data[i].values[j];
        }

        file >> data[i].humus;
    }

    file.close();

    // Build a decision tree
    Node* root = build_tree(data, 50);

    // Normalize the data
    vector < double>  test_data(8);

    // ������� ������ ������ �����
    test_data = { 1373,	1464,	1773,	1674,	2253,	3548,	3812,	4122 };

    double predicted_humus = predict(root, test_data);
    cout << "Predicted humus value: " << predicted_humus << endl;

    return 0;
}