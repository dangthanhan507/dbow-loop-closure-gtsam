#include <iostream>
#include <vector>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>


using namespace DBoW2;
using namespace std;

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void loadFeatures(vector<vector<cv::Mat > > &features, std::string filepath, unsigned nimages);
void testDatabase(const vector<vector<cv::Mat > > &features, const int nimages_eval);

const int NIMAGES_EVAL = 1000;

int main()
{
  vector<vector<cv::Mat > > features2;
  loadFeatures(features2, "images3", NIMAGES_EVAL);

  testDatabase(features2, NIMAGES_EVAL);

  return 0;
}

void loadFeatures(vector<vector<cv::Mat > > &features, std::string filepath, unsigned nimages)
{
  features.clear();
  features.reserve(nimages);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;
  for(unsigned i = 0; i < nimages; ++i)
  {
    stringstream ss;
    ss << filepath << "/" << "image" << i << ".jpg";
    //print the image path

    cv::Mat image = cv::imread(ss.str());
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    //check fi it loaded the image
    if(image.empty())
    {
      cerr << "Couldn't read image: " << ss.str() << endl;
      exit(1);
    }

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    features.push_back(vector<cv::Mat >());
    changeStructure(descriptors, features.back());
  }
}

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}


void testDatabase(const vector<vector<cv::Mat > > &features, const int nimages_eval)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  OrbVocabulary voc("small_voc.yml.gz");
  
  OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < nimages_eval; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < nimages_eval; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("db_eval.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  cout << "Retrieving database once again..." << endl;
  OrbDatabase db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}