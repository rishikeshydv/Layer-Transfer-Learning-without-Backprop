export default function Home() {
  return (
    <div className="bg-[#F1E9DA]">
      <div className="bg-[#576066] h-16 flex items-center justify-center text-5xl font-serif">
        Youtube Video Recommender
      </div>
      <div className="bg-[#F1E9DA] h-screen mt-40 mb-10 text-black font-serif ">
        <h1 className="flex justify-center text-3xl font-bold">
          View your next YT Video
        </h1>
        <br />
        <br />
        <br />
        <div className="flex items-center">
          <button className="bg-green-500 text-white py-2 px-4 rounded mr-4 ml-4">
            Current Video Title:
          </button>
          <input
            type="text"
            className="border rounded h-32 w-full py-2 px-4 mr-5"
            placeholder="Enter text..."
          />
        </div>
        <br />
        <br />
        <br />

        <div className="flex items-center">
          <button className="bg-green-500 text-white py-2 px-4 rounded mr-4 ml-4">
            Current Video Tag:
          </button>
          <input
            type="text"
            className="border rounded h-32 w-full py-2 px-4 mr-5"
            placeholder="Enter text..."
          />
        </div>
        <br />
        <br />
        <br />

        <div className="flex items-center">
          <button className="bg-green-500 text-white py-2 px-4 rounded mr-4 ml-4">
            Recipe Fed:
          </button>
          <input
            type="text"
            className="border rounded h-32 w-full py-2 px-4 mr-5"
            placeholder="Enter text..."
          />
        </div>
        <br />
        <br />
        <br />
        <div className="flex items-center">
          <button className="bg-green-500 text-white py-2 px-4 rounded mr-4 ml-4">
            Namespace Created:
          </button>
          <input
            type="text"
            className="border rounded h-32 w-full py-2 px-4 mr-5"
            placeholder="Enter text..."
          />
        </div>
        <br />
        <br />
        <br />
        <div className="flex items-center">
          <button className="bg-green-500 text-white py-2 px-4 rounded mr-4 ml-4">
            Next Video Title:
          </button>
          <input
            type="text"
            className="border rounded h-32 w-full py-2 px-4 mr-5"
            placeholder="Enter text..."
          />
        </div>
        <br />
        <br />
        <br />
        <div className="flex items-center">
          <button className="bg-green-500 text-white py-2 px-4 rounded mr-4 ml-4">
            Next Video Tag:
          </button>
          <input
            type="text"
            className="border rounded h-32 w-full py-2 px-4 mr-5"
            placeholder="Enter text..."
          />
        </div>
        <br />
        <br />
        <br />
      </div>
    </div>
  );
}
