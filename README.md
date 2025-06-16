# Customer-Segmentation-for-Financial-Product
Analysis and segmentation of financial clients using PCA and KMeans clustering to identify profiles and behavioral patterns. Includes visualizations and application of the model to new data to validate results in Python.


This project aims to segment the customers of a financial company using unsupervised learning techniques. To do so, the project begins by importing and visually exploring the original dataset, which includes variables such as salary, credit card limit, length of service, credit type, and whether they were offered a product. This initial exploration allows for the identification of potential visual groupings, particularly by observing relationships such as salary vs. credit limit.

Principal component analysis (PCA) was then applied, a statistical technique that reduces the size of the dataset without losing too much information. This facilitates subsequent analysis and improves model performance. PCA was able to reduce the number of variables while retaining 80% of the total variance. With the transformed data, the elbow method was used to determine the optimal number of clusters, which indicated that 5 was the appropriate number to segment customers.

The KMeans algorithm was then applied with 5 clusters, grouping customers according to similarities found in the PCA-transformed data. The quality of the model was subsequently assessed using metrics such as the Silhouette Score, Calinski-Harabasz, and Davies-Bouldin, confirming that the clusters were well-defined and separated. In addition, 3D graphs were created to visualize these groups more intuitively.

Once the groups were created, their average characteristics (salary, credit limit, length of time as a customer) were analyzed, and the number of products offered or specific types of credit was counted. This allowed the clusters to be characterized and the typical profile of customers in each group to be understood. Bar charts were also generated to clearly visualize these differences and compare them.

Finally, the same trained model was applied to a new dataset with 50 new customers. These data were processed using the same scaler, PCA, and clustering model, ensuring consistency with the original model. They were then visualized alongside previous customers in a 3D graph, clearly differentiating new customers through the use of different symbols. Average salaries, credit limits, and tenure were also calculated for new customers based on the cluster to which they were assigned.

In summary, this project demonstrates how unsupervised learning can be used to efficiently segment customers. Furthermore, a reusable system was built that allows new customers to be classified in real time, facilitating strategic decision-making based on well-defined profiles.

This project aims to segment the customers of a financial company using unsupervised learning techniques. To do so, the project begins with the import and visual exploration of the original dataset, which includes variables such as salary, credit card limit, customer tenure, credit type, and whether they were offered a product. This initial exploration allows potential visual groupings to be identified, particularly by observing relationships such as salary versus credit limit.

Principal component analysis (PCA) was then applied, a statistical technique that reduces the dimension of the dataset without losing too much information. This facilitates subsequent analysis and improves model performance. PCA managed to reduce the number of variables while retaining 80% of the total variance. With the transformed data, the elbow method was used to determine the optimal number of clusters, which indicated that 5 was the appropriate number to segment customers.

The KMeans algorithm was then applied with 5 clusters, grouping customers according to similarities found in the PCA-transformed data. The quality of the model was subsequently evaluated using metrics such as the Silhouette Score, Calinski-Harabasz, and Davies-Bouldin, confirming that the clusters were well-defined and separated. In addition, 3D graphs were created to visualize these groups more intuitively.

Once the groups were created, their average characteristics (salary, credit limit, length of time as a customer) were analyzed, and the number of customers offered specific products or types of credit was counted. This allowed the clusters to be characterized and the typical customer profile in each group to be understood. Bar charts were also generated to clearly visualize these differences and compare them.

Finally, the same trained model was applied to a new dataset with 50 new customers. These data were processed using the same scaler, PCA, and clustering model, ensuring consistency with the original model. They were then visualized alongside the previous customers in a 3D graph, clearly differentiating the new ones through the use of different symbols. The data was also calculated the average salaries, credit limits, and seniority for new customers based on the cluster to which they were assigned.

In short, this project demonstrates how unsupervised learning can be used to efficiently segment customers. Furthermore, a reusable system was built that allows new customers to be classified in real time, facilitating strategic decision-making based on well-defined profiles.
