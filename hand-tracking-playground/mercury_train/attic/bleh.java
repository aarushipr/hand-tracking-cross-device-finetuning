/**
 * Calculates the euclidean distance from a point to a line segment.
 *
 * @param v     the point
 * @param a     start of line segment
 * @param b     end of line segment 
 * @return      distance from v to line segment [a,b]
 *
 * @author      Afonso Santos
 */
 public static
 double
 distanceToSegment( final R3 v, final R3 a, final R3 b )
 {
   final R3 ab  = b.sub( a ) ;
   final R3 av  = v.sub( a ) ;

   if (av.dot(ab) <= 0.0)           // Point is lagging behind start of the segment, so perpendicular distance is not viable.
     return av.modulus( ) ;         // Use distance to start of segment instead.

   final R3 bv  = v.sub( b ) ;

   if (bv.dot(ab) >= 0.0)           // Point is advanced past the end of the segment, so perpendicular distance is not viable.
     return bv.modulus( ) ;         // Use distance to end of the segment instead.

   return (ab.cross( av )).modulus() / ab.modulus() ;       // Perpendicular distance of point to segment.
}