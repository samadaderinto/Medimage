import * as React from "react";
import { SVGProps } from "react";
const UploadFileIcon = (props: SVGProps<SVGSVGElement>) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    xmlnsXlink="http://www.w3.org/1999/xlink"
    fill="none"
    width={45}
    height={31.5}
    viewBox="0 0 45 31.5"
    {...props}
  >
    <g transform="matrix(1,0,0,-1,0,63)">
      <path
        d="M10.125,31.5Q5.83594,31.640625,2.95312,34.45312Q0.140625,37.33594,0,41.625Q0.0703125,45,1.89844,47.5312Q3.72656,50.0625,6.75,51.1875Q6.75,51.4688,6.75,51.75Q6.89062,56.5312,10.0547,59.6953Q13.2188,62.8594,18,63Q21.1641,62.9297,23.6953,61.4531Q26.2266,59.9062,27.7734,57.375Q29.3906,58.5,31.5,58.5Q34.3828,58.4297,36.2812,56.5312Q38.1797,54.6328,38.25,51.75Q38.25,50.4844,37.8281,49.289100000000005Q40.9219,48.6562,42.9609,46.1953Q44.9297,43.8047,45,40.5Q44.9297,36.70312,42.3984,34.10156Q39.7969,31.5703125,36,31.5L10.125,31.5ZM15.6797,46.7578Q14.6953,45.5625,15.6797,44.3672Q16.875,43.3828,18.0703,44.3672L20.8125,47.1094L20.8125,37.6875Q20.9531,36.14062,22.5,36Q24.0469,36.14062,24.1875,37.6875L24.1875,47.1094L26.9297,44.3672Q28.125,43.3828,29.3203,44.3672Q30.3047,45.5625,29.3203,46.7578L23.6953,52.3828Q22.5,53.3672,21.3047,52.3828L15.6797,46.7578Z"
        fill="#000000"
        fillOpacity={1}
        style={
          {
            //   mixBlendMode: "passthrough",
          }
        }
      />
    </g>
  </svg>
);
export default UploadFileIcon;
