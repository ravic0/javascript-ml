// import _ from 'lodash';


const dropPoints = [];
// const dropReference = 300; // Should not be fixed
// const k = 3;

// function distance(testPoint, trainingPosition) { // Individual variable
//   return Math.abs(testPoint - trainingPosition);
// }

function distance(trainingPoint,testPoint){
// const dropPoints = Math.pow(trainingPoint[0]-testPoint[0],2);
// const bouncePoints = Math.pow(trainingPoint[1]-testPoint[1],2);
// const sizePoints = Math.pow(trainingPoint[2]-testPoint[2],2); // Switch to genericness from 3 independent variables
// let value = 0;

// for(let i=0;i<trainingPoint.length-1;i++){ // -1 to avoid last value i.e label
//   value += Math.pow((trainingPoint[i]-testPoint[i]),2);
// }

// return Math.sqrt(value); //Switch to a better syntax

return _.chain(trainingPoint)
.zip(testPoint) // Ex: A = [1,1] , B = [4,5] ==  O/P [[1,4],[1,5]]
.map(([a,b]) => Math.pow((a-b),2)) // a is the first element of zip [1,4] and b is [1,5]
//NOTE THAT a-b will work for even more elements than 2 i.e 1-4=-3, 1-5=-4 and so on.
.sum()
.value()
} 

function generateDataset(dataset, number)
{
  const data = _.shuffle(dataset); // To avoid sequential data which would act as bad data for train and test
  const testData = _.slice(data, 0, number);
  const trainingData = _.slice(data, number);
  return [testData, trainingData]; //[test, training]
}

function knn(trainingData, testPoint, k=1)
{
  // console.log("Testing for k: ", testPoint);
  return _.chain(trainingData)
    // .map(row => [distance(row[0],testPoint), row[3]]) 
    .map(row => {
      // const result = _.last(row); //Label
      // // distance(row[0],testPoint) // individual variable
      // // distance(row[0],row[1],testPoint[0],testPoint[1]); // two variables
      // const dist = distance(row,testPoint); // genericness
      // return [dist,result]; // Switch to better syntax
      
      //_.initial returns everything apart from the last element
      return [distance(_.initial(row),testPoint), _.last(row)] //testPoint will ALWAYS have 3 values as per runAnalysis
    })
    .sortBy(row => row[0])
    .slice(0, k)
    .countBy(row => row[1]) // creates obj
    .toPairs() // creates array
    .sortBy(row => row[1]) // most appearing bucket
    .last()
    .first()
    .parseInt()
    .value();
}

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  dropPoints.push([dropPosition, bounciness,size, bucketLabel]);
}

function runAnalysis() {
  const testSize = 100;
  const k = 10;
  // const [testData, trainingData] = generateDataset(minMax(dropPoints, 3), testSize);
  // let rightPoints = 0;
  // for(let i=0; i<testData.length; i++)
  // {
  // const actualPoint = testData[i][0];
  // const actualBucket = testData[i][3];
  // const probableBucket = knn(trainingData, actualPoint);

  // if(probableBucket === actualBucket)
  // rightPoints++;

  // // console.log("Actual bucket",testData[i][3], "  Bucket :", probableBucket);
  // } // Switch to a better syntax

  
  // _.range(1,kRange).forEach((k) => { // Decided that best value of k was 10
  _.range(0,3).forEach((feature) => { // Checking contribution of each feature towards the result. 3 is non inclusive
    const data = _.map(dropPoints, row => [row[feature], _.last(row)]);
    const [testData, trainingData] = generateDataset(minMax(data, 1), testSize); // Moved inside to check effectiveness of each feature
    
    const accuracy = _.chain(testData)
    .filter(row => knn(trainingData, _.initial(row), k) === _.last(row)) // We will assume testData (row) is passed without label 
    .size()
    .divide(testSize)
    .value();
  
    console.log("Feature: ", feature  ," Accuracy is: ",accuracy*100);
  })

  // console.log("Accuracy: ", (accuracy.length/testSize)*100,"%" );
  // console.log(probableBucket);
}

function minMax(dataset, featureCount) {  // featureCount helps avoid last value i.e label
const deepClone = _.cloneDeep(dataset); // To not mutate original dataset

for(let i=0; i<featureCount; i++) {
  const column = deepClone.map(row => row[i]); // Normalize one feature at a time
  const min = _.min(column);
  const max = _.max(column);

  for(let j=0; j<deepClone.length; j++)
    deepClone[j][i] = (deepClone[j][i] - min) / (max - min);
}

return deepClone;
} 

