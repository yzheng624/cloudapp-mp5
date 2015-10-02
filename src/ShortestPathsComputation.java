import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.conf.LongConfOption;
import org.apache.giraph.edge.Edge;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;

import java.io.IOException;

/**
 * Compute shortest paths from a given source.
 */
public class ShortestPathsComputation extends BasicComputation<
    IntWritable, IntWritable, NullWritable, IntWritable> {
  /** The shortest paths id */
  public static final LongConfOption SOURCE_ID =
      new LongConfOption("SimpleShortestPathsVertex.sourceId", 1,
          "The shortest paths id");

  /**
   * Is this vertex the source id?
   *
   * @param vertex Vertex
   * @return True if the source id
   */
  private boolean isSource(Vertex<IntWritable, ?, ?> vertex) {
    return vertex.getId().get() == SOURCE_ID.get(getConf());
  }

  @Override
  public void compute(
      Vertex<IntWritable, IntWritable, NullWritable> vertex,
      Iterable<IntWritable> messages) throws IOException {
      int currentComponent = vertex.getValue().get();

      if (getSuperstep() == 0) {

          if (isSource(vertex)) {
              vertex.setValue(new IntWritable(0));
              for (Edge<IntWritable, NullWritable> edge : vertex.getEdges()) {
                  IntWritable neighbor = edge.getTargetVertexId();
                  sendMessage(neighbor, vertex.getValue());
              }
          } else {
              vertex.setValue(new IntWritable(Integer.MAX_VALUE));
          }

          vertex.voteToHalt();
          return;
      }

      boolean changed = false;
      for (IntWritable message : messages) {
        int candidateComponent = message.get() + 1;
        if (candidateComponent < currentComponent) {
          currentComponent = candidateComponent;
          changed = true;
        }
      }

      if (changed) {
        vertex.setValue(new IntWritable(currentComponent));
        sendMessageToAllEdges(vertex, vertex.getValue());
      }
      vertex.voteToHalt();
  }
}
