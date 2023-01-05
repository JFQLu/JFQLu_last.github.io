### Reachable Vertices in Directed Graph using GraphX

This project was conducted as coursework in Big Data Management (COMP9313) at UNSW. The problem states:

"Given a directed graph, for each vertex v, compute the number of vertices that are reachable from v in the graph (including v itself if there is a path starting from v and ending at v). For example, for node 0, the number of vertices that are reachable from 0 is 6, since there exists a path from node 0 to each node in the graph."

This project requires the use of the Apache Spark big data processing framework, specifically, GraphX which provides an API for graph (vertices and edges) processing.

GraphX extends the Spark RDD (Resilient Distributed Dataset) API to support graph processing by providing a set of graph-specific RDDs, which are distributed collections of data that can be processed in parallel. These include VertexRDD, EdgeRDD, and Graph.

The Pregel operator will be important in solving this problem. 
The Pregel operator performs the computation in a series of supersteps, with each superstep consisting of the following steps:

1. Each vertex receives the messages sent to it in the previous superstep and executes the vertex program, possibly updating its data and sending new messages to its neighbors.
2. The messages are collected and stored in a message buffer.
3. The vertex program of each vertex is executed again, this time using the updated message buffer.

This process is repeated until the algorithm converges or a maximum number of supersteps is reached.

To solve our problem above we start by importing the required libraries.

```scala
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.graphx._
```

Next, we must define a vertex program. This is a user-defined function that is executed on each vertex in the graph. It takes as input the vertex ID, the current vertex data, and a message received by the vertex in the previous superstep, and it returns an updated vertex data. Here we have the vertex program take the message (newData) and merge it with the vertex data (origData) removing duplicates, i.e. we take the union of newData and origData.  

```scala
// Vertex Program: merges vertex data (List[VertexId]) with message (List[VertexId]) removing duplicates 
def vertexProgram(id: VertexId, origData: List[VertexId], newData: List[VertexId]) : List[VertexId] = ( origData ::: newData ).distinct
```

We also define a mergeMsg function, this has no access to the context of any Vertex -- it just takes individual messages and creates a single message which is then sent to the vertex program as newData. Here we simply take the union of all messages. 

```scala
def mergeMsg(list1: List[VertexId], list2: List[VertexId]) : List[VertexId] = (list1 ::: list2).distinct
```

Finally, we define a sendMsg function, it takes as input the source and destination vertices of the edge, and it returns a message to be sent from the source vertex to the destination vertex. Here we first check if the vertex is reachable from itself. We also check if there are any messages that need to be added to srcAttr. 

```scala
def sendMsg(triplet: EdgeTriplet[List[VertexId],Double]) : Iterator[(VertexId, List[VertexId])] = {
    // Append destination vertexId with destination vertexAttribute (List[VertexId]) and save to new val
    val newList = (triplet.dstId :: triplet.dstAttr).distinct 
        
    // Note that we are sending messages to the source vertex from the destination vertex (ie. we "flip" the direction of edges in the graph)
    // If source vertex is also reachable from destination vertex then the source node can reach itself
    if (triplet.srcAttr == triplet.dstAttr) {                       
        Iterator((triplet.srcId, newList))  
    // If the srcAttr list is the same length as the newList formed through recursion, then there is nothing new to add to the srcAttr list
    } else if (triplet.srcAttr.intersect(newList).length != newList.length) {   
        Iterator((triplet.srcId, newList))      
    } else {
        Iterator.empty 
    }
}
```

Now we can create the main function:

```scala
def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName("GraphX reachable vertices").setMaster("local")
    val sc = new SparkContext(conf)
    val inputFile = args(0)
    val outputFile = args(1)

    val edges = sc.textFile(inputFile)
    // Create the graph
    val edgelist = edges.map(x => x.split(" ")).map(x=> Edge(x(1).toLong, x(2).toLong, 1d))
    val graph = Graph.fromEdges[Double, Double](edgelist, 0.0)
    // Initialize each vertex attribute to empty list (List[VertexId]())
    val initialGraph = graph.mapVertices((id, _) => List[VertexId]())
    val result = initialGraph
                .pregel(
                    initialMsg = List.empty[VertexId],
                    activeDirection = EdgeDirection.Out  
                )(vertexProgram, sendMsg, mergeMsg)
                .mapVertices((_, neighbors) => neighbors.length).vertices.sortBy(_._1).filter(t => t._2 != Double.PositiveInfinity).map(a => a._1 + ":" + a._2).saveAsTextFile(outputFile)
}
```
Note that input is in the format "EdgeId FromNodeId ToNodeId" and output in the format "VertexId:Number_of_reachable_vertices"

#### Takeaway: 

From this project we see the power of GraphX for graph processing applications. It is useful for a wide range of applications that involve graph data, such as social network analysis, recommendation systems, and fraud detection.












