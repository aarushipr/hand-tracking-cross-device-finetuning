
int YOLO_GRID_X = 13;
int YOLO_GRID_Y = 13;
int YOLO_NUM_BB = 5;





typedef struct
{
  float x;
  float y;
  float w;
  float h;
} Box;

typedef struct detection
{
  Box bbox;
  float conf;
  int class_id;
  float prob;
} detection;

typedef struct
{
 float dx;
 float dy;
 float dw;
 float dh;
} Dbox;

Box FloatToBox(const float fx, const float fy, const float fw, const float fh)
{
  Box b;
  b.x = fx;
  b.y = fy;
  b.w = fw;
  b.h = fh;

  return b;
}

float Overlap(float x1, float w1, float x2, float w2)
{
  float l1 = x1 - w1 / 2;
  float l2 = x2 - w2 / 2;
  float left = l1 > l2 ? l1 : l2;

  float r1 = x1 + w1 / 2;
  float r2 = x2 + w2 / 2;
  float right = r1 < r2 ? r1 : r2;

  return right - left;
}

float BoxIntersection(const Box& a, const Box& b)
{
  float w = Overlap(a.x, a.w, b.x, b.w);
  float h = Overlap(a.y, a.h, b.y, b.h);

  if(w < 0 || h < 0)
    return 0;

  return w * h;
}

float BoxUnion(const Box& a, const Box& b)
{
  return a.w * a.h + b.w * b.h - BoxIntersection(a, b);
}

float BoxIOU(const Box& a, const Box& b)
{
  return BoxIntersection(a, b) / BoxUnion(a, b);
}

void FilterBoxesNMS(std::vector<detection>& det, int nBoxes, float th_nms)
{
  int count = nBoxes;
  for (size_t i = 0;i < count; ++i)
  {
    Box a = det[i].bbox;
    for (size_t j = 0; j < count; ++j)
    {
      if (i == j) continue;
      if (det[i].class_id != det[j].class_id) continue;

      Box b = det[j].bbox;
      float b_intersection = BoxIntersection(a, b);
      if (BoxIOU(a, b) > th_nms ||
          b_intersection >= a.h * a.w - 1 ||
          b_intersection >= b.h * b.w - 1)
      {
        if (det[i].prob > det[j].prob)
        {
          det[j].prob = 0;
        }
        else
        {
          det[i].prob = 0;
        }
      }
    }
  }
}

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
  return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
  os << "[";
  for (int i = 0; i < v.size(); ++i)
  {
      os << v[i];
      if (i != v.size() - 1)
      {
          os << ", ";
      }
  }
  os << "]";

  return os;
}

std::vector<std::string> ReadLabels(const std::string& labelsFile)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelsFile);
    while (std::getline(fp, line))
        labels.push_back(line);

    return labels;
}