import * as React from "react";
import { SVGProps } from "react";
const SuccessfulUploadsIcon = (props: SVGProps<SVGSVGElement>) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    xmlnsXlink="http://www.w3.org/1999/xlink"
    fill="none"
    width={38}
    height={48}
    viewBox="0 0 38 48"
    {...props}
  >
    <defs>
      <clipPath id="master_svg0_2_047">
        <rect x={12} y={15} width={14} height={16} rx={0} />
      </clipPath>
    </defs>
    <g>
      <rect
        x={0}
        y={0}
        width={38}
        height={48}
        rx={19}
        fill="#DCFCE7"
        fillOpacity={1}
        style={
          {
            //   mixBlendMode: "passthrough",
          }
        }
      />
      <g clipPath="url(#master_svg0_2_047)">
        <g transform="matrix(1,0,0,-1,0,56.6875)">
          <g>
            <path
              d="M25.7188,38.0625Q26,37.75,26,37.34375Q26,36.9375,25.7188,36.625L17.71875,28.625Q17.40625,28.34375,17,28.34375Q16.59375,28.34375,16.28125,28.625L12.28125,32.625Q12,32.9375,12,33.34375Q12,33.75,12.28125,34.0625Q12.59375,34.34375,13,34.34375Q13.40625,34.34375,13.71875,34.0625L17,30.75L24.2812,38.0625Q24.5938,38.34375,25,38.34375Q25.4062,38.34375,25.7188,38.0625Z"
              fill="#16A34A"
              fillOpacity={1}
              style={
                {
                  // mixBlendMode: "passthrough",
                }
              }
            />
          </g>
        </g>
      </g>
    </g>
  </svg>
);
export default SuccessfulUploadsIcon;
