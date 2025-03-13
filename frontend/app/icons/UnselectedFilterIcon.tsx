import * as React from "react";
import { SVGProps } from "react";
const UnselectedFilter = (props: SVGProps<SVGSVGElement>) => (
  <svg
    width={20}
    height={20}
    viewBox="0 0 20 20"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    {...props}
  >
    <rect
      x={1}
      y={1}
      width={18}
      height={18}
      rx={4}
      stroke="#0066FF"
      strokeWidth={1.5}
    />
  </svg>
);
export default UnselectedFilter;
