#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/cc/framework/ops.h>

#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/graph.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/kernels/summary_interface.h>
#include <tensorflow/contrib/tensorboard/db/summary_file_writer.h>

using namespace tensorflow;

int main()
{
  int size = 100;
  std::vector<tensorflow::Tensor> output;
  
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ClientSession session(root);

  GraphDef graph;
  root.ToGraphDef(&graph);
  graph::SetDefaultDevice("tpu-demo", &graph);

  auto A = tensorflow::Tensor(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({size}));
  auto Amap = A.tensor<float, 1>();

  auto B = tensorflow::Tensor(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({size}));
  auto Bmap = B.tensor<float, 1>();

  auto C = tensorflow::Tensor(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({size}));
  auto Cmap = C.tensor<float, 1>();

  for (int i = 0; i < size; i++){
    Amap(i) = (i*1.0f + 4.)*2.5;
    Bmap(i) = (i*1.0f + 5.)*2.5;
    Cmap(i) = (i*1.0f + 6.)*0.1;
  }

  auto add = tensorflow::ops::Add(root, A, B);
  auto mul = tensorflow::ops::Multiply(root, 2.4f, C);
  auto sub = tensorflow::ops::Subtract(root, add, mul);

  TF_CHECK_OK(session.Run({sub}, &output));

  root.ToGraphDef(&graph);
  SummaryWriterInterface* w;
  TF_CHECK_OK(CreateSummaryFileWriter(1, 0, ".", ".tpugraph", Env::Default(), &w));
  TF_CHECK_OK(w->WriteGraph(0, std::make_unique<GraphDef>(graph)));

  return 0;
}
